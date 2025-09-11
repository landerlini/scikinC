"""
A minimalistic fixture registry for pytest that enables:
- Centralized tracking of fixtures per test module
- Automatic parametrization of tests over registered fixtures
- Cleaner test definitions with granular reporting


"""
import sys
import pytest


class FixtureRegistry:
    """
    A registry to track pytest fixtures per module and generate
    parametrized tests automatically.
    """

    def __init__(self):
        self._fixtures_per_module = dict()

    def register(self, *test_categories):
        """
        Decorator to register a pytest fixture and store its name
        under the calling module.

        Args:
            *test_categories: strings identifying sets of fixtures to be tested separately.
                Tests always belong to the default group named after the module itself.
                Note that name sets are visible at global scope, it is client's responsibility to ensure uniqueness
                of the set names (if intended).

        Returns:
            function: The same function wrapped as a pytest fixture
        """
        test_categories = [sys._getframe(1).f_globals["__name__"], *test_categories]

        def decorator(f):
            fixture_name = f.__name__

            for test_category in test_categories:
                # Initialize fixture set for the module if needed
                self._fixtures_per_module.setdefault(test_category, set())

                # Register the fixture name
                self._fixtures_per_module[test_category].add(fixture_name)

            # Return the input function as a fixture
            return pytest.fixture(f)

        return decorator


    def test(self, test_category=None):
        """
        Decorator to generate a parametrized test over all registered
        fixtures in the calling module. Each fixture will be passed
        to the test function individually.

        Args:
            test_category: a string defining the set of tests to run. Defaults to callers' module name

        Returns:
            function: A parametrized test function
        """
        if test_category is None:
            test_category = sys._getframe(1).f_globals["__name__"]

        def decorator(f):
            fixture_names = self._fixtures_per_module.get(test_category, [])

            @pytest.mark.parametrize ('fixture_name', fixture_names)
            def test_forward(fixture_name, request):
                fixture_value = request.getfixturevalue(fixture_name)
                return f(fixture_value)

            return test_forward

        return decorator

# Singleton instance
fixtures = FixtureRegistry()






