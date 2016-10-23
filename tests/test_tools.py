from pydnn import tools
import unittest
import tempfile
import shutil
import os

class TestFileOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tempDir = tempfile.mkdtemp()
        cls._subdirs = ['a', 'B', '123']
        cls._files = ['f1.file', 'f2.file', 'f3.file']
        cls._files_in_first = ["1.file", "2.file"]
        for d in cls._subdirs:
            os.mkdir(os.path.join(cls._tempDir, d))
        for f in cls._files:
            open(os.path.join(cls._tempDir, f), "a").close()
        for f in cls._files_in_first:
            open(os.path.join(cls._tempDir, cls._subdirs[0], f), "a").close()
        print(cls._tempDir)
    
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tempDir)

    def test_get_sub_dirs(self):
        self.assertEqual(
            set(tools.get_sub_dirs(self._tempDir, False)),
            set(os.path.join(self._tempDir, subdir) for subdir in self._subdirs))
        self.assertEqual(
            set(tools.get_sub_dirs(self._tempDir, True)),
            set(self._subdirs))

    def test_get_files(self):
        self.assertEqual(            
            set(tools.get_files(self._tempDir, False)),
            set(os.path.join(self._tempDir, f) for f in self._files))
        self.assertEqual(            
            set(tools.get_files(self._tempDir, True)),
            set(self._files))