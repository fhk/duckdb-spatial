#include "spatial/geometry/spatial_index_interface.hpp"
#include "spatial/geometry/cluster_types.hpp"
#include "spatial/geometry/flat_rtree.hpp"
#include "spatial/geometry/dbscan_engine.hpp"
#include "spatial/modules/main/spatial_functions.hpp"
#include "spatial/spatial_types.hpp"
#include "spatial/util/function_builder.hpp"

#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

namespace duckdb {
namespace {

struct DBSCANWindowState {
	std::vector<int32_t> cluster_ids;
	std::vector<bool> is_null;
};

struct ST_ClusterDBSCAN_Point2D {
	static idx_t StateSize(const AggregateFunction &) {
		return sizeof(DBSCANWindowState);
	}

	static void StateInitialize(const AggregateFunction &, data_ptr_t state) {
		new (state) DBSCANWindowState();
	}

	static void StateDestructor(Vector &state, AggregateInputData &, idx_t count) {
		UnifiedVectorFormat sdata;
		state.ToUnifiedFormat(count, sdata);
		auto states = UnifiedVectorFormat::GetData<DBSCANWindowState *>(sdata);
		for (idx_t i = 0; i < count; i++) {
			auto idx = sdata.sel->get_index(i);
			if (sdata.validity.RowIsValid(idx)) {
				states[idx]->~DBSCANWindowState();
			}
		}
	}

	static void WindowInit(AggregateInputData &, const WindowPartitionInput &partition, data_ptr_t g_state) {
		auto &wstate = *reinterpret_cast<DBSCANWindowState *>(g_state);
		const idx_t row_count = partition.count;
		wstate.cluster_ids.assign(row_count, -1);
		wstate.is_null.assign(row_count, false);

		if (row_count == 0 || !partition.inputs || partition.column_ids.size() < 3) {
			return;
		}

		std::vector<spatial::Point2D> valid_points;
		std::vector<size_t> row_mapping; // maps valid_points index -> partition row index
		valid_points.reserve(row_count);
		row_mapping.reserve(row_count);

		double eps = 0.0;
		int64_t min_points = 1;
		bool params_read = false;

		idx_t current_row = 0;
		for (auto &chunk : partition.inputs->Chunks(partition.column_ids)) {
			const idx_t chunk_size = chunk.size();
			if (chunk_size == 0) {
				continue;
			}

			auto &pt_vec = chunk.data[0];
			auto &eps_vec = chunk.data[1];
			auto &min_pts_vec = chunk.data[2];

			if (!params_read) {
				auto eps_val = eps_vec.GetValue(0);
				auto min_pts_val = min_pts_vec.GetValue(0);
				if (!eps_val.IsNull()) {
					eps = eps_val.GetValue<double>();
				}
				if (!min_pts_val.IsNull()) {
					min_points = min_pts_val.GetValue<int64_t>();
				}
				params_read = true;
			}

			pt_vec.Flatten(chunk_size);
			auto &entries = StructVector::GetEntries(pt_vec);
			entries[0]->Flatten(chunk_size);
			entries[1]->Flatten(chunk_size);

			auto x_data = FlatVector::GetData<double>(*entries[0]);
			auto y_data = FlatVector::GetData<double>(*entries[1]);
			auto &pt_validity = FlatVector::Validity(pt_vec);
			auto &x_validity = FlatVector::Validity(*entries[0]);
			auto &y_validity = FlatVector::Validity(*entries[1]);

			for (idx_t i = 0; i < chunk_size; ++i) {
				const idx_t global_row = current_row + i;

				if (!pt_validity.RowIsValid(i) || !x_validity.RowIsValid(i) || !y_validity.RowIsValid(i)) {
					wstate.is_null[global_row] = true;
					continue;
				}

				valid_points.emplace_back(x_data[i], y_data[i]);
				row_mapping.push_back(global_row);
			}

			current_row += chunk_size;
		}

		if (valid_points.empty() || eps <= 0.0 || min_points <= 0) {
			return;
		}

		spatial::FlatRTree2D rtree(32);
		rtree.Build(spatial::ArrayView<spatial::Point2D>(valid_points));

		spatial::DBSCANParams params(eps, min_points);
		auto result =
		    spatial::DBSCANEngine::Cluster2D(spatial::ArrayView<spatial::Point2D>(valid_points), rtree, params);

		for (size_t i = 0; i < valid_points.size(); ++i) {
			const size_t orig_row = row_mapping[i];
			wstate.cluster_ids[orig_row] = result.GetClusterId(i);
		}
	}

	static void Window(AggregateInputData &, const WindowPartitionInput &partition, const_data_ptr_t g_state,
	                   data_ptr_t, const SubFrames &, Vector &result, idx_t rid) {
		auto &wstate = *reinterpret_cast<const DBSCANWindowState *>(g_state);
		const auto global_row = partition.row_index;
		if (global_row >= wstate.cluster_ids.size() || wstate.is_null[global_row] ||
		    wstate.cluster_ids[global_row] < 0) {
			FlatVector::SetNull(result, rid, true);
		} else {
			FlatVector::GetData<int32_t>(result)[rid] = wstate.cluster_ids[global_row];
		}
	}
};

} // namespace

void RegisterSpatialWindowFunctions(ExtensionLoader &loader) {
	// Register ST_ClusterDBSCAN for POINT_2D
	AggregateFunction cluster_point2d(
	    "ST_ClusterDBSCAN", {GeoTypes::POINT_2D(), LogicalType::DOUBLE, LogicalType::BIGINT}, LogicalType::INTEGER,
	    ST_ClusterDBSCAN_Point2D::StateSize, ST_ClusterDBSCAN_Point2D::StateInitialize,
	    nullptr, // update (null for window-only aggregate)
	    nullptr, // combine
	    nullptr, // finalize
	    nullptr  // simple_update
	);

	cluster_point2d.destructor = ST_ClusterDBSCAN_Point2D::StateDestructor;
	cluster_point2d.window_init = ST_ClusterDBSCAN_Point2D::WindowInit;
	cluster_point2d.window = ST_ClusterDBSCAN_Point2D::Window;

	FunctionBuilder::RegisterAggregate(loader, "ST_ClusterDBSCAN", [&](AggregateFunctionBuilder &func) {
		func.SetFunction(cluster_point2d);
		func.SetDescription("Performs DBSCAN density-based clustering over 2D points using an inbuilt R-Tree.");
		func.SetExample("SELECT id, ST_ClusterDBSCAN(pt, 0.5, 5) OVER () AS cid FROM points;");
		func.CanThrowErrors();
		func.SetTag("ext", "spatial");
		func.SetTag("category", "clustering");
	});
}

} // namespace duckdb
