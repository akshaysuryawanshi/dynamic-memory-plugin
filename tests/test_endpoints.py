"""The small HTTP surface and native-call ownership after a waiter leaves."""
import asyncio
from types import SimpleNamespace as NS
import unittest

from fastapi import FastAPI, HTTPException
import httpx
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler

from dynamic_memory_plugin import endpoints


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.cancelled = False

        async def utility(method, *args):
            self.calls.append((method, args))
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return {'operation': {'state': 'draining'}, 'capacity': {}}

        async def health():
            pass

        self.app = FastAPI()
        self.app.state.args = NS(log_error_stack=False)
        init_exception_handler(self.app)
        @self.app.get('/native-error')
        async def native_error():
            raise HTTPException(418, 'native error')
        self.app.state.dynamic_memory_client = NS(engine_core=NS(call_utility_async=utility), check_health=health)
        endpoints.DynamicMemoryEndpoints().attach_router(self.app)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')

    async def asyncTearDown(self):
        self.release.set()
        await asyncio.sleep(.01)
        await self.client.aclose()

    async def test_resize_needs_only_target_and_strategy(self):
        response = await self.client.post('/v1/dynamic-memory/resize', json={'target_bytes': 256})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.calls, [('dynamic_memory_resize', (256, 'drain', None))])
        routes = {route.path for route in self.app.routes if route.path.startswith('/v1/dynamic-memory')}
        self.assertEqual(routes, {'/v1/dynamic-memory/status', '/v1/dynamic-memory/resize'})

    async def test_validation_cannot_dispatch_and_native_errors_remain_native(self):
        for body in ({'target_bytes': True}, {'target_bytes': '256'}, {'target_bytes': -1},
                     {'target_bytes': 256, 'request_id': 'removed'},
                     {'target_bytes': 256, 'strategy': 'unknown'}):
            with self.subTest(body=body):
                response = await self.client.post('/v1/dynamic-memory/resize', json=body)
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.calls, [])
        response = await self.client.get('/native-error')
        self.assertEqual(response.status_code, 418)
        self.assertEqual(response.json()['error']['message'], 'native error')

    async def test_http_cancellation_does_not_cancel_engine_mutation(self):
        self.release.clear()
        waiter = asyncio.create_task(self.client.post('/v1/dynamic-memory/resize', json={'target_bytes': 256}))
        await asyncio.wait_for(self.entered.wait(), 1)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertFalse(self.cancelled)
        registry = self.app.state.dynamic_memory_utility_tasks
        self.assertEqual(len(registry.tasks), 1)
        self.release.set()
        await asyncio.wait_for(asyncio.gather(*tuple(registry.tasks)), 1)
        await asyncio.sleep(0)
        self.assertFalse(registry.tasks)

    async def test_waiter_timeout_retains_owned_call(self):
        self.release.clear()
        # A caller's timeout does not establish engine cancellation or rollback.
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(self.client.post(
                '/v1/dynamic-memory/resize', json={'target_bytes': 256}), .02)
        self.assertFalse(self.cancelled)
        registry = self.app.state.dynamic_memory_utility_tasks
        self.assertEqual(len(registry.tasks), 1)
