from index import Dog
import pytest

class Test_Dog:

    @pytest.fixture()
    def dog(self):
        """
        Returns a Dog object.

        Args:
            self: write your description
        """
        return Dog("Dado")

    class Test_Woof:
        

        def test_woof_1(self, dog):
            """
            Test woof for 1 dog.

            Args:
                self: write your description
                dog: write your description
            """
            result = dog.woof(1)
            assert result == 1

        def test_woof_2(self, dog):
            """
            Test the woof of a dog.

            Args:
                self: write your description
                dog: write your description
            """
            result = dog.woof(2)
            assert result == 2

    class Test_Bark:

        def test_bark_1(self, dog):
            """
            Test the bark function for dog.

            Args:
                self: write your description
                dog: write your description
            """
            result = dog.bark(1)
            assert result == 1

        def test_bark_2(self, dog):
            """
            Test the bark function for 2 dogs.

            Args:
                self: write your description
                dog: write your description
            """
            result = dog.bark(2)
            assert result == 2