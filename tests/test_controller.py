"""Failure boundaries that remain necessary without the administration layer."""
from concurrent.futures import Future
import unittest

from dynamic_memory_plugin.controller import ResizeController


class Backend:
    def __init__(self, outcome='applied'):
        self.events = []
        self.bytes = 512
        self.paused = False
        self.ready = False
        self.future = Future()
        self.outcome = outcome

    def capacity(self):
        return {'allocated_kv_bytes': self.bytes, 'num_blocks': self.bytes // 16,
                'paused': self.paused}

    def prepare(self, target, strategy):
        self.events.append('prepare')
        return {'old_bytes': self.bytes, 'new_bytes': target}

    def pause(self):
        self.paused = True
        self.events.append('pause')
        return self.future

    def assert_drained(self):
        assert self.paused
        self.events.append('drained')

    def replace(self, plan):
        self.events.append('replace')
        if self.outcome == 'lost_response':
            raise RuntimeError('Worker response lost after possible mutation')
        return {'outcome': self.outcome, 'error': None}

    replace_preserving = replace

    def publish(self, plan, rollback):
        self.events.append('rollback' if rollback else 'publish')
        self.bytes = plan['old_bytes'] if rollback else plan['new_bytes']

    def publish_preserving(self, plan):
        self.publish(plan, False)

    def resume(self):
        self.events.append('resume')
        self.paused = False

    def fence(self, error):
        self.events.append('fence')

    def ensure_paused(self):
        self.paused = True

    def scheduling_paused(self):
        return self.paused

    def begin_preserving(self, plan, callback):
        self.progress = callback
        self.events.append('withdraw')

    def preserving_ready(self):
        return self.ready

    def end_preserving(self, cancel=False, failed=False):
        self.events.append('failed' if failed else 'cancel' if cancel else 'end')
        if failed:
            self.ensure_paused()


class ResizeTests(unittest.TestCase):
    def test_drain_defers_mutation_and_rejects_overlap(self):
        backend = Backend()
        controller = ResizeController(backend)
        self.assertEqual(controller.resize(256)['operation']['state'], 'draining')
        with self.assertRaises((ValueError, RuntimeError)):
            controller.resize(768)
        self.assertEqual(backend.events, ['prepare', 'pause'])
        backend.future.set_result(None)
        self.assertEqual(backend.events, ['prepare', 'pause', 'drained', 'replace', 'publish', 'resume'])
        self.assertEqual(controller.status()['capacity']['allocated_kv_bytes'], 256)

    def test_rollback_and_preflight_rejection_have_distinct_publication(self):
        for outcome, publication in [('rolled_back', ['rollback']), ('rejected', [])]:
            with self.subTest(outcome=outcome):
                backend = Backend(outcome)
                backend.future.set_result(None)
                result = ResizeController(backend).resize(768)
                self.assertEqual(result['operation']['state'], outcome)
                self.assertEqual(backend.events, ['prepare', 'pause', 'drained', 'replace', *publication, 'resume'])
                self.assertEqual(backend.bytes, 512)

    def test_unknown_worker_outcome_fences_and_does_not_claim_capacity(self):
        for strategy in ('drain', 'preserve'):
            with self.subTest(strategy=strategy):
                backend = Backend('lost_response')
                backend.ready = True
                backend.future.set_result(None)
                controller = ResizeController(backend)
                result = controller.resize(256, strategy=strategy)
                self.assertTrue(result['restart_required'])
                self.assertIsNone(result['capacity']['allocated_kv_bytes'])
                self.assertIn('fence', backend.events)
                self.assertNotIn('resume', backend.events)
                self.assertNotIn('publish', backend.events)
                with self.assertRaises((ValueError, RuntimeError)):
                    controller.resize(512)

    def test_failed_drain_never_reaches_storage(self):
        backend = Backend()
        controller = ResizeController(backend)
        controller.resize(256)
        backend.future.set_exception(RuntimeError('Drain failed'))
        self.assertTrue(controller.status()['restart_required'])
        self.assertNotIn('replace', backend.events)

    def test_preserve_waits_for_boundary_and_status_does_not_mutate(self):
        backend = Backend()
        controller = ResizeController(backend)
        controller.resize(256, strategy='preserve')
        backend.ready = True
        self.assertEqual(controller.status()['operation']['state'], 'withdrawing')
        self.assertNotIn('replace', backend.events)
        backend.progress()
        self.assertEqual(backend.events, ['prepare', 'withdraw', 'replace', 'publish', 'end'])
        self.assertEqual(controller.status()['operation']['state'], 'completed')

    def test_withdrawal_timeout_restores_old_capacity_without_mutation(self):
        backend = Backend()
        controller = ResizeController(backend)
        controller.resize(256, strategy='preserve', withdrawal_timeout_seconds=1)
        controller.deadline = 0
        result = controller.status()
        self.assertEqual(result['operation']['state'], 'timed_out')
        self.assertEqual(backend.bytes, 512)
        self.assertEqual(backend.events, ['prepare', 'withdraw', 'cancel'])
        self.assertFalse(result['restart_required'])

    def test_invalid_inputs_do_not_prepare_storage(self):
        for target in (True, 0, -1, '256', 1.5):
            with self.subTest(target=target):
                backend = Backend()
                with self.assertRaises((ValueError, RuntimeError)):
                    ResizeController(backend).resize(target)
                self.assertEqual(backend.events, [])

    def test_same_size_does_not_discard_cache(self):
        backend = Backend()
        result = ResizeController(backend).resize(512)
        self.assertEqual(result['operation']['state'], 'unchanged')
        self.assertEqual(backend.events, ['prepare'])
