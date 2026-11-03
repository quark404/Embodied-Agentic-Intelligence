import unittest
import socket
import urllib.error
from unittest.mock import MagicMock, patch

from cluster_eval.network import BYPASS, proxy_environment, validate_proxy_environment
from cluster_eval.submit_preflight import job_environment
from cluster_eval.storage import BOOTSTRAP_WHEEL_URL, network_error, network_inventory, probe_network_url


class ClusterNetworkTests(unittest.TestCase):
    def test_network_errors_distinguish_transport_without_disclosing_urls_or_credentials(self):
        for reason in (TimeoutError(), socket.gaierror(-2, 'dns secret'),
                       OSError('Tunnel connection failed: 407 secret-token')):
            with self.subTest(reason=type(reason).__name__):
                result = network_error(urllib.error.URLError(reason))
                self.assertEqual(result['reason_type'], type(reason).__name__)
                self.assertNotIn('secret', str(result))
                self.assertFalse(result['reachable'])
        self.assertEqual(result['proxy_tunnel_status'], 407)

    def test_http_denial_is_distinct_from_a_connection_timeout(self):
        error = urllib.error.HTTPError('https://example.invalid', 403, 'denied', {}, None)
        with patch('cluster_eval.storage.urllib.request.urlopen', side_effect=error):
            result = probe_network_url('https://example.invalid')
        self.assertTrue(result['reachable'])
        self.assertEqual(result['status'], 403)

    def test_diagnostic_compares_direct_path_and_only_connects_to_the_configured_proxy(self):
        proxy = 'http://10.0.0.2:18000'
        connection = MagicMock()
        with patch('cluster_eval.storage.probe_network_url', return_value={'reachable': False}) as probe, \
                patch('cluster_eval.storage.urllib.request.getproxies', return_value={'https': proxy}), \
                patch('cluster_eval.storage.socket.create_connection', return_value=connection) as connect:
            result = network_inventory()['_diagnostics']
        connect.assert_called_once_with(('10.0.0.2', 18000), timeout=5)
        probe.assert_any_call('https://pypi.org/simple/', direct=True)
        probe.assert_any_call('https://pypi.hobot.cc/simple/', direct=True)
        probe.assert_any_call(BOOTSTRAP_WHEEL_URL)
        self.assertTrue(result['proxy_tcp']['reachable'])
        self.assertEqual(result['effective_proxy_host'], '10.0.0.2')

    def test_explicit_proxy_is_credential_free_and_does_not_replace_local_traffic(self):
        environment = proxy_environment('http://10.0.0.2:18000')
        validate_proxy_environment(environment)
        self.assertEqual(environment['HTTP_PROXY'], environment['https_proxy'])
        self.assertEqual(environment['NO_PROXY'], BYPASS)
        self.assertIn('127.0.0.1', BYPASS)

    def test_rejects_secret_bearing_loopback_and_ambiguous_proxy_urls(self):
        for value in ('http://user:secret@proxy:8080', 'http://proxy:8080/?token=secret',
                      'http://proxy:8080/#secret', 'http://proxy:8080/path', 'http://proxy',
                      'socks5://proxy:1080', 'http://localhost:8080', 'http://127.0.0.1:8080',
                      'http://proxy:70000', 'http://proxy:8080 bad'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                proxy_environment(value)

    def test_only_complete_consistent_proxy_settings_extend_the_job_allowlist(self):
        base = {'EAI_CLUSTER_CONFIRMED': '0', 'EAI_RUN_ID': 'test', 'EAI_RESULT_DIR': '/private',
                'EAI_ORIGIN_SHA256': 'digest', 'PYTHONDONTWRITEBYTECODE': '1', 'EAI_STORAGE_MODE': 'job_local'}
        network = proxy_environment('http://10.0.0.2:18000')
        result = job_environment({'env': {**base, **network}}, 'private-token')
        self.assertEqual(result['EAI_CLUSTER_CONFIRMED'], '1')
        self.assertNotIn('private-token', str(result))
        for altered in ({'https_proxy': network['https_proxy']},
                        {**network, 'HTTP_PROXY': 'http://other:8080'},
                        {**network, 'NO_PROXY': '*'}):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                job_environment({'env': {**base, **altered}}, 'private-token')


if __name__ == '__main__':
    unittest.main()
