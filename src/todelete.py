from index import Dog
import pytest

class Test_Dog:

    @pytest.fixture()
    def dog(self):
        """
        Returns the notification.

        Args:
            self: (todo): write your description
        """
        return Dog("Dado")

    class Test_Woof:
        

        def test_woof_1(self, dog):
            """
            Test if two sets of the same.

            Args:
                self: (todo): write your description
                dog: (array): write your description
            """
            result = dog.woof(1)
            assert result == 1

        def test_woof_2(self, dog):
            """
            Convenience function to a test.

            Args:
                self: (todo): write your description
                dog: (todo): write your description
            """
            result = dog.woof(2)
            assert result == 2

    class Test_Bark:

        def test_bark_1(self, dog):
            """
            Test if two test sets are equal.

            Args:
                self: (todo): write your description
                dog: (todo): write your description
            """
            result = dog.bark(1)
            assert result == 1

        def test_bark_2(self, dog):
            """
            Test if the two - qubits of - two - qubits.

            Args:
                self: (todo): write your description
                dog: (todo): write your description
            """
            result = dog.bark(2)
            assert result == 2