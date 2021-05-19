from index import Dog
import pytest

class Test_Dog:

    @pytest.fixture()
    def dog(self):
        return Dog("Dado")

    class Test_Woof:
        

        def test_woof_1(self, dog):
            result = dog.woof(1)
            assert result == 1

        def test_woof_2(self, dog):
            result = dog.woof(2)
            assert result == 2

    class Test_Bark:

        def test_bark_1(self, dog):
            result = dog.bark(1)
            assert result == 1

        def test_bark_2(self, dog):
            result = dog.bark(2)
            assert result == 2