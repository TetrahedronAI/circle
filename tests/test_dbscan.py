import unittest

import tests.helpers as _
from src.circleml.cluster import DBScanCla
from src.circleml.datasets import make_clusters


class TestDBScan(unittest.TestCase):
    def test_dbscan(self):
        X, y = make_clusters()
        dbscan = DBScanCla()
        dbscan.fit_predict(X)

    def test_call(self):
        X, y = make_clusters()
        dbscan = DBScanCla()
        dbscan(X)