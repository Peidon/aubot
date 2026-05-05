import unittest
from clustering import semantic_cluster
from text_processor import extract_primary_meaning

class RecognizerTests(unittest.TestCase):
    def setUp(self):
        # self.recognizer = Recognizer()
        self.primary_mean_test_cases = [
            ['job title', 'work experience', 'job title', 'work experience'],
            ['school name', 'education',  'school name', 'school or university', 'education', 'education', 'delete'],
            ['search', 'education',  'field of study', 'field of study'],
            ['search', 'skills', 'skills', 'type', 'to', 'add'],
            ['education',  'first year attended date section year', 'input', 'year', 'from', 'current value is 2020'],
            ['grade average', 'education', 'grade average', 'overall result  gpa', 'field of study', 'degree', 'school or university', 'education', 'education', 'delete'],
            ['work experience', 'start date', 'date section month', 'input', 'month', 'from', 'current', 'value', 'is'],
            ['work experience', 'start date', 'date section year', 'input', 'year', 'from', 'current', 'value', 'is'],
            ['work experience', 'end date', 'date section month', 'input', 'month', 'to', 'current', 'value', 'is'],
            ['work experience', 'end date', 'date section year', 'input', 'year', 'to', 'current', 'value', 'is'],
            ['are you awaiting the hearing of charges in a civil or criminal court of law', 'yes', 'no', 'question 34354'],
            ['work experience', 'role description', 'role', 'description', 'description', 'from', 'i', 'currently', 'work', 'here', 'location', 'work', 'experience', 'experience']
        ]

    def test_extract_primary_meaning(self):
        for phrases in self.primary_mean_test_cases:
            result = extract_primary_meaning(phrases)
            print(f"{phrases}\n→ {result}\n")




class ClusteringTestCase(unittest.TestCase):

    def test_clustering(self):
        # Example 1: Job-related fields
        print("=" * 60)
        print("Example 1: Job-related fields")
        print("=" * 60)

        job_fields = [
            'work experience',
            'role description',
            'location',
            'job duties',
            'address',
            'responsibilities',
            'workplace',
            'career history'
        ]

        clusters = semantic_cluster(
            job_fields,
            distance_threshold=0.4,
            model="sentence-transformers"
        )

        self.assertTrue(len(clusters)>0)

        print("\n" + "=" * 60)
        print("Example 2: Mixed categories")
        print("=" * 60)

        mixed_fields = [
            'email address',
            'phone number',
            'contact info',
            'job title',
            'position',
            'salary',
            'compensation',
            'benefits package',
            'mobile number'
        ]

        clusters2 = semantic_cluster(
            mixed_fields,
            distance_threshold=0.45,
            model="sentence-transformers"
        )

        self.assertTrue(len(clusters2) > 0)

        print("\n" + "=" * 60)
        print("Example 3: With fixed number of clusters")
        print("=" * 60)

        role_description = [
            'work experience',
            'role description',
            'role',
            'description', 'from', 'i', 'currently', 'work', 'here', 'location', 'experience']

        clusters3 = semantic_cluster(
            role_description,
            n_clusters=2,
            model="sentence-transformers"
        )

        self.assertTrue(len(clusters3) > 0)


if __name__ == '__main__':
    unittest.main()
