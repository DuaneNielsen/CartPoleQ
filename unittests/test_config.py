from unittest import TestCase
from mentality import Config

class TestConfig(TestCase):
    def test_config_loading(self):
        config = Config()
        config = Config()
        print(config.config['last_run_id'])

