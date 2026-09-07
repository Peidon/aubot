import unittest
from text_processor import select_representative, Recognizer

class RecognizerTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = Recognizer()
        self.primary_mean_test_cases = [
            ['school name', 'education', 'school or university'],
            ['education', 'field of study'],
            ['skills', 'type to add skills'],
            ['job title', 'work experience'],
            ['company name', 'work experience', 'company'],
            ['work experience', 'start date', 'date section month', 'input', 'month', 'from', 'current value is mm','yyyy'],
            ['work experience', 'start date', 'date section year', 'input', 'year', 'from', 'current value is mm','yyyy'],
            ['work experience', 'end date', 'date section month', 'input', 'month', 'to', 'current value is mm','yyyy'],
            ['work experience', 'end date', 'date section year', 'input', 'year', 'to', 'current value is mm', 'yyyy'],
            ['work experience', 'role description', 'role description'],
            ['candidate mobile', 'formatted number', 'formatted number'],
            ['candidate mobile', 'normalized number', 'normalized number'],
            ['candidate mobile', 'country code', 'country code']
        ]

    def test_extract_representative(self):
        result = select_representative(self.primary_mean_test_cases)
        for represent in result:
            print(f"{represent}")

    def test_similarities(self):
        source = ["what is your expected annual salary", "what is your current annual salary", "What are your remuneration expectations in relation to this role"]
        target = ["what are your salary expectations"]
        result = self.recognizer.similarities(source, target)
        print(result)


if __name__ == '__main__':
    unittest.main()
