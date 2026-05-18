import unittest
from clustering import semantic_cluster
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
            'career history',
            'skills', 'type', 'to', 'add'
        ]

        clusters = semantic_cluster(
            job_fields,
            distance_threshold=0.7,
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
            distance_threshold=0.7,
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
