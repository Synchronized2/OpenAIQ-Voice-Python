import unittest
from unittest.mock import patch

import weather


class WeatherHelpersTests(unittest.TestCase):
    def test_regular_question_does_not_trigger(self):
        self.assertFalse(weather.is_weather_query("给我讲个笑话"))
        self.assertIsNone(weather.lookup_weather("给我讲个笑话"))

    def test_city_extraction(self):
        self.assertEqual(weather.extract_city("明天上海天气怎么样"), "上海")
        self.assertEqual(weather.extract_city("what is the weather in London"), "London")

    def test_network_failure_is_safe(self):
        with patch.object(weather, "_http_json", side_effect=OSError("offline")):
            result = weather.lookup_weather("北京今天天气怎么样", timeout=0.1)
        self.assertIn("暂时不可用", result)


if __name__ == "__main__":
    unittest.main()
