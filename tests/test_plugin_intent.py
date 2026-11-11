"""Loosely worded plugin requests (ogun_core.plugin_intent): new build vs. update, and which plugin is meant."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ogun_core.plugin_intent import match_plugin, resolve_plugin_request  # noqa: E402


def _plugins():
    d = Path(tempfile.mkdtemp())
    (d / "__init__.py").write_text("")
    (d / "roman_numeral.py").write_text('description = "Converts between integers and Roman numerals."\n')
    (d / "temperature_converter.py").write_text('description = "Convert temperatures between Celsius and Fahrenheit."\n')
    (d / "date_difference.py").write_text('    description: str = "Number of days between two dates."\n')
    return d


def test_vague_new_plugin_requests_are_builds():
    d = _plugins()
    for text in ("can you make a plugin that tells me how many days are between two dates",
                 "make a temperature converter plugin",
                 "i need something that turns a title into a url slug, as a plugin",
                 "Build a plugin named dice_roller that rolls dice"):
        assert resolve_plugin_request(text, d) == ("build", None), text


def test_vague_change_requests_find_the_installed_plugin():
    d = _plugins()
    assert resolve_plugin_request("the roman numeral plugin should accept lowercase too", d) == ("update", "roman_numeral")
    assert resolve_plugin_request("the plugin that counts days between dates should also understand dates like "
                                  "03/01/2024", d) == ("update", "date_difference")
    assert resolve_plugin_request("can the temperature plugin do kelvin as well", d) == ("update",
                                                                                         "temperature_converter")


def test_other_requests_are_left_alone():
    d = _plugins()
    for text in ("what is the capital of france", "build a command-line tool that counts lines", "",
                 "thanks, that works"):
        assert resolve_plugin_request(text, d) is None, text


def test_no_guess_on_a_tie():
    d = Path(tempfile.mkdtemp())
    (d / "word_count.py").write_text('description = "Count words."\n')
    (d / "word_list.py").write_text('description = "List words."\n')
    assert match_plugin("the word plugin should ignore case", {"word_count": "Count words.",
                                                                "word_list": "List words."}) is None


def test_round_two_wording():
    d = Path(tempfile.mkdtemp())
    assert resolve_plugin_request("give yourself the ability to count vowels in some text", d) == ("build", None)
    assert resolve_plugin_request("how do i learn to swim", d) is None
    (d / "palindrome_checker.py").write_text('description = "Check whether a string is a palindrome."\n')
    (d / "bmi_calculator.py").write_text('description = "Calculate body mass index from weight and height."\n')
    assert resolve_plugin_request("the palindrome thing should ignore spaces and punctuation", d) == (
        "update", "palindrome_checker")
    assert resolve_plugin_request("make the bmi one also say if it's underweight or obese", d) == (
        "update", "bmi_calculator")
    assert resolve_plugin_request("what is a palindrome", d) is None


def test_bare_plugin_descriptions_are_builds():
    d = Path(tempfile.mkdtemp())
    assert resolve_plugin_request("plugin that tells me if a year is a leap year", d) == ("build", None)
    assert resolve_plugin_request("can you whip up a plugin to reverse the words in a sentence", d) == ("build", None)
    assert resolve_plugin_request("teach yourself to convert miles to kilometers", d) == ("build", None)
    assert resolve_plugin_request("which plugin for dates do you have", d) is None
    assert resolve_plugin_request("what plugins do you have", d) is None


def test_change_to_a_missing_plugin_builds_it():
    d = Path(tempfile.mkdtemp())
    assert resolve_plugin_request("the miles plugin should round to 2 decimal places", d) == ("build", None)
    assert resolve_plugin_request("does the miles plugin round its results", d) is None
    assert resolve_plugin_request("is the miles plugin installed", d) is None


def test_generic_name_words_do_not_hide_the_plugin():
    """Live: 'the miles plugin' missed miles_to_kilometers_converter and a second plugin was built."""
    d = Path(tempfile.mkdtemp())
    (d / "miles_to_kilometers_converter.py").write_text('description = "Convert miles to kilometers."\n')
    (d / "bmi_calculator.py").write_text('description = "Body mass index."\n')
    assert resolve_plugin_request("the miles plugin should round to 2 decimal places", d) == (
        "update", "miles_to_kilometers_converter")


def test_installer_rejects_a_plugin_without_the_basic_shape():
    """Live: a chat-model tool call installed `def execute(self, miles): return miles * 1.6` and called it tested."""
    from ogun_core.plugin_manager import plugin_shape_problems as shape
    bad = "from ogun_core.base_tool import BaseTool\nclass M(BaseTool):\n    def execute(self, miles):\n        return miles * 1.6\n"
    assert len(shape(bad)) == 3
    no_params = ("from ogun_core.base_tool import BaseTool, ToolResult\nclass T(BaseTool):\n    name = 't'\n"
                 "    def execute(self, **kwargs):\n        return ToolResult.success('ok')\n")
    assert shape(no_params) == []
    reads = no_params.replace("ToolResult.success('ok')", "ToolResult.success(kwargs.get('x'))")
    assert len(shape(reads)) == 1 and "parameters_schema" in shape(reads)[0]
