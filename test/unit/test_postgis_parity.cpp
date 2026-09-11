#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include "spatial/geometry/flat_rtree.hpp"
#include "spatial/geometry/dbscan_engine.hpp"

using namespace duckdb::spatial;

void TestCancellation() {
	std::vector<Point2D> points(10000, Point2D(0, 0));
	size_t checks = 0;
	bool cancel = true;
	FlatRTree2D tree(32, [&]() {
		if (cancel && ++checks == 3) {
			throw std::runtime_error("cancelled");
		}
	});
	bool interrupted = false;
	try {
		tree.Build(points);
	} catch (const std::runtime_error &) {
		interrupted = true;
	}
	assert(interrupted);
	// A cancelled build can be rebuilt, and cancellation also reaches queries.
	cancel = false;
	tree.Build(points);
	cancel = true;
	checks = 0;
	interrupted = false;
	try {
		DBSCANEngine::Cluster2D(tree, DBSCANParams(1, 2));
	} catch (const std::runtime_error &) {
		interrupted = true;
	}
	assert(interrupted);
}

void TestCoordinateRanges() {
	for (double scale : {1e-200, 1.0, 1e200}) {
		std::vector<Point2D> points = {{0, 0}, {scale, scale}};
		FlatRTree2D tree;
		tree.Build(points);
		// A square candidate box must still receive an exact circular check.
		const auto result = DBSCANEngine::Cluster2D(tree, DBSCANParams(scale, 2));
		assert(result.NumNoise() == 2);
	}
	for (size_t node_size : {0, 1}) {
		bool rejected = false;
		try {
			FlatRTree2D tree(node_size);
		} catch (const std::invalid_argument &) {
			rejected = true;
		}
		assert(rejected);
	}
}

void TestSmallTreeRegression() {
	std::vector<Point2D> points = {{0.0, 0.0}, {0.1, 0.0}, {0.0, 0.1}, {5.0, 5.0}, {5.1, 5.0}, {2.5, 2.5}};
	FlatRTree2D tree(32); // Same node size as the SQL window function.
	tree.Build(points);
	auto result = DBSCANEngine::Cluster2D(tree, DBSCANParams(0.2, 2));
	assert(result.GetClusterIds() == std::vector<int32_t>({0, 0, 0, 1, 1, -1}));

	// Compare neighborhood searches with brute force across the small-tree
	// cutoff and multiple tree levels. Reuse the tree to check rebuilds too.
	for (size_t count : {0, 1, 2, 31, 32, 33, 1024, 1025, 2, 0}) {
		points.clear();
		for (size_t i = 0; i < count; ++i) {
			points.emplace_back(double(i % 17), double(i / 17));
		}
		tree.Build(points);
		assert(tree.Count() == count);
		std::vector<size_t> actual = {9999};
		tree.RadiusSearch(Point2D(-100, -100), 0.2, actual);
		assert(actual.empty());
		for (const auto &center : points) {
			std::vector<size_t> expected;
			for (size_t i = 0; i < count; ++i) {
				if (std::hypot(center.x - points[i].x, center.y - points[i].y) <= 1.25) {
					expected.push_back(i);
				}
			}
			tree.RadiusSearch(center, 1.25, actual);
			std::sort(actual.begin(), actual.end());
			assert(actual == expected);
		}
	}
	std::cout << "Small-tree DBSCAN and radius-search regressions passed!" << std::endl;
}

void TestPostGISDocExample() {
	std::cout << "[PostGIS Parity] Testing PostGIS Documentation Example..." << std::endl;
	// PostGIS docs:
	// A1: ST_MakeEnvelope(0, 0, 10, 10)     -> center: (5, 5)
	// A2: ST_MakeEnvelope(30, 0, 40, 10)    -> center: (35, 5)
	// B1: ST_MakeEnvelope(100, 0, 110, 10)  -> center: (105, 5)
	// B2: ST_MakeEnvelope(130, 0, 140, 10)  -> center: (135, 5)
	// noise: ST_MakeEnvelope(250, 0, 260, 10) -> center: (255, 5)
	// eps = 50, minpoints = 2
	// Expected:
	// Cluster 0: A1, A2
	// Cluster 1: B1, B2
	// Noise (NULL): noise
	std::vector<Point2D> points = {
	    Point2D(5.0, 5.0),   // 0: A1
	    Point2D(35.0, 5.0),  // 1: A2
	    Point2D(105.0, 5.0), // 2: B1
	    Point2D(135.0, 5.0), // 3: B2
	    Point2D(255.0, 5.0)  // 4: noise
	};

	FlatRTree2D rtree(2);
	rtree.Build(points);

	DBSCANParams params(50.0, 2);
	DBSCANResult result = DBSCANEngine::Cluster2D(rtree, params);

	assert(result.NumClusters() == 2);
	assert(result.NumNoise() == 1);

	// Verify A1 and A2 are in the same cluster (cluster 0)
	assert(result.GetClusterId(0) == 0);
	assert(result.GetClusterId(1) == 0);

	// Verify B1 and B2 are in the same cluster (cluster 1)
	assert(result.GetClusterId(2) == 1);
	assert(result.GetClusterId(3) == 1);

	// Verify noise point is unclustered (cluster ID = -1, which maps to SQL NULL)
	assert(result.GetClusterId(4) == -1);
	assert(result.IsNoise(4));

	std::cout << "  -> PostGIS Doc Example PASSED! (Clusters: {A1, A2} => 0, {B1, B2} => 1, {noise} => NULL)"
	          << std::endl;
}

void TestPostGISRegressionT101() {
	std::cout << "[PostGIS Parity] Testing PostGIS Regression Test t101 (eps=0.8, minpoints=1)..." << std::endl;
	// Table dbscan_inputs:
	// 1: POINT (0 0)
	// 2: POINT (0 1)
	// 3: POINT (-0.5 0.5)
	// 4: POINT (1 0)
	// 5: POINT (1 1)
	// 6: POINT (1.0 0.5)
	// With minpoints=1, all points must be clustered (equivalent to connected components)
	// Expected:
	// Left group {1, 2, 3} -> Cluster 0
	// Right group {4, 5, 6} -> Cluster 1
	std::vector<Point2D> points = {
	    Point2D(0.0, 0.0),  // 0 (id 1)
	    Point2D(0.0, 1.0),  // 1 (id 2)
	    Point2D(-0.5, 0.5), // 2 (id 3)
	    Point2D(1.0, 0.0),  // 3 (id 4)
	    Point2D(1.0, 1.0),  // 4 (id 5)
	    Point2D(1.0, 0.5)   // 5 (id 6)
	};

	FlatRTree2D rtree(2);
	rtree.Build(points);

	DBSCANParams params(0.8, 1);
	DBSCANResult result = DBSCANEngine::Cluster2D(rtree, params);

	assert(result.NumClusters() == 2);
	assert(result.NumNoise() == 0);

	assert(result.GetClusterId(0) == 0);
	assert(result.GetClusterId(1) == 0);
	assert(result.GetClusterId(2) == 0);

	assert(result.GetClusterId(3) == 1);
	assert(result.GetClusterId(4) == 1);
	assert(result.GetClusterId(5) == 1);

	std::cout << "  -> PostGIS t101 PASSED! (Left: 0, Right: 1, 0 noise)" << std::endl;
}

void TestPostGISRegressionT102() {
	std::cout << "[PostGIS Parity] Testing PostGIS Regression Test t102 (eps=0.8, minpoints=4)..." << std::endl;
	// Each group has only 3 points. Since minpoints=4, neither group can form a cluster!
	// Expected: All 6 points are noise (cluster ID = -1, maps to SQL NULL)
	std::vector<Point2D> points = {Point2D(0.0, 0.0), Point2D(0.0, 1.0), Point2D(-0.5, 0.5),
	                               Point2D(1.0, 0.0), Point2D(1.0, 1.0), Point2D(1.0, 0.5)};

	FlatRTree2D rtree(2);
	rtree.Build(points);

	DBSCANParams params(0.8, 4);
	DBSCANResult result = DBSCANEngine::Cluster2D(rtree, params);

	assert(result.NumClusters() == 0);
	assert(result.NumNoise() == 6);

	for (size_t i = 0; i < 6; ++i) {
		assert(result.GetClusterId(i) == -1);
		assert(result.IsNoise(i));
	}

	std::cout << "  -> PostGIS t102 PASSED! (All 6 points are NULL/noise)" << std::endl;
}

void TestPostGISRegressionT103() {
	std::cout << "[PostGIS Parity] Testing PostGIS Regression Test t103 (eps=0.6, minpoints=3)..." << std::endl;
	// Left group: distance is ~0.7071 > 0.6. Each point has < 3 neighbors within 0.6.
	// Right group: point (1.0, 0.5) has (1.0, 0.0) at dist 0.5 and (1.0, 1.0) at dist 0.5.
	// It has 3 points within 0.6 (including itself). Core point!
	// Points 4 and 5 are border points.
	// Expected:
	// Left group {1, 2, 3} -> NULL (noise)
	// Right group {4, 5, 6} -> Cluster 0
	std::vector<Point2D> points = {
	    Point2D(0.0, 0.0),  // 0
	    Point2D(0.0, 1.0),  // 1
	    Point2D(-0.5, 0.5), // 2
	    Point2D(1.0, 0.0),  // 3
	    Point2D(1.0, 1.0),  // 4
	    Point2D(1.0, 0.5)   // 5
	};

	FlatRTree2D rtree(2);
	rtree.Build(points);

	DBSCANParams params(0.6, 3);
	DBSCANResult result = DBSCANEngine::Cluster2D(rtree, params);

	assert(result.NumClusters() == 1);
	assert(result.NumNoise() == 3);

	// Left group are NULL
	assert(result.GetClusterId(0) == -1);
	assert(result.GetClusterId(1) == -1);
	assert(result.GetClusterId(2) == -1);
	assert(result.IsNoise(0));
	assert(result.IsNoise(1));
	assert(result.IsNoise(2));

	// Right group forms Cluster 0
	assert(result.GetClusterId(3) == 0);
	assert(result.GetClusterId(4) == 0);
	assert(result.GetClusterId(5) == 0);

	std::cout << "  -> PostGIS t103 PASSED! (Left: NULL, Right: Cluster 0 with 1 Core + 2 Border points)" << std::endl;
}

void TestPostGISRegressionSinglePoint3612b() {
	std::cout << "[PostGIS Parity] Testing PostGIS Regression Test #3612b (Single point with minpoints=5)..."
	          << std::endl;
	// PostGIS query: SELECT '#3612b', ST_ClusterDBSCAN(ST_Point(1,1), 20.1, 5) OVER();
	// Expected: NULL (single point cannot satisfy min_points = 5)
	std::vector<Point2D> points = {Point2D(1.0, 1.0)};

	FlatRTree2D rtree(2);
	rtree.Build(points);

	DBSCANParams params(20.1, 5);
	DBSCANResult result = DBSCANEngine::Cluster2D(rtree, params);

	assert(result.NumClusters() == 0);
	assert(result.NumNoise() == 1);
	assert(result.GetClusterId(0) == -1);
	assert(result.IsNoise(0));

	std::cout << "  -> PostGIS #3612b PASSED! (Single point => NULL/noise)" << std::endl;
}

int main() {
	TestCancellation();
	TestCoordinateRanges();
	TestSmallTreeRegression();
	std::cout << "==========================================================" << std::endl;
	std::cout << "      PostGIS ST_ClusterDBSCAN Parity & Reproduction      " << std::endl;
	std::cout << "==========================================================" << std::endl;

	TestPostGISDocExample();
	TestPostGISRegressionT101();
	TestPostGISRegressionT102();
	TestPostGISRegressionT103();
	TestPostGISRegressionSinglePoint3612b();

	std::cout << std::endl;
	std::cout << "All PostGIS ST_ClusterDBSCAN reproduction tests passed with 100% parity!" << std::endl;
	return 0;
}
