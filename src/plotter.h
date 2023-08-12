#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "basic_bench.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

void make_plots(const std::vector<bench_data>& dataset, std::string prefix) {
	// Ensure the directories exist before writing the plots to files
	std::filesystem::create_directories(
			std::filesystem::path(prefix).parent_path());

	std::map<std::string, std::vector<bench_data>> entries_by_name;
	for (const auto& bd : dataset) {
		entries_by_name[bd.engine_name].push_back(bd);
	}

	double pointsize = 10.0;

	// querytime-recall
	plt::figure_size(1280, 720);
	for (const auto& [name, bdv] : entries_by_name) {
		std::vector<double> tpq, recall;
		for (const auto& bd : bdv) {
			tpq.push_back(bd.time_per_query_ns);
			recall.push_back(bd.recall);
		}
		// plt::clear();
		plt::scatter(recall, tpq, pointsize, {{"label", name}});
		// plt::title(name + "_recall-querytime");
		// plt::save("./plots/" + name + ".png");
	}
	plt::xlabel("recall");
	plt::ylabel("querytime(ns)");
	// plt::xscale("log");
	// plt::xticks(std::vector<float>(
	// 		{0.0, 0.5, 1.0 - 1.0e-1, 1.0 - 1.0e-3, 1.0 - 1.0e-4, 1.0}));
	plt::yscale("log");
	plt::legend();
	plt::title("recall-querytime for k-NN");
	plt::save(prefix + "time-recall.png");

	// querytime-avgdist
	plt::figure_size(1280, 720);
	for (const auto& [name, bdv] : entries_by_name) {
		std::vector<double> tpq, avgdist;
		for (const auto& bd : bdv) {
			tpq.push_back(bd.time_per_query_ns);
			avgdist.push_back(bd.average_distance);
		}
		plt::scatter(avgdist, tpq, pointsize, {{"label", name}});
	}
	plt::xlabel("avgdist");
	plt::ylabel("querytime(ns)");
	plt::xscale("log");
	plt::yscale("log");
	plt::legend();
	plt::title("avgdist-querytime for 1-NN");
	plt::save(prefix + "time-avgdist.png");

	// buildtime-recall
	plt::figure_size(1280, 720);
	for (const auto& [name, bdv] : entries_by_name) {
		std::vector<double> y, x;
		for (const auto& bd : bdv) {
			x.push_back(bd.time_to_build_ns);
			y.push_back(bd.recall);
		}
		plt::scatter(x, y, pointsize, {{"label", name}});
	}
	plt::xlabel("time_to_build(ns)");
	plt::ylabel("recall");
	plt::xscale("log");
	plt::yscale("log");
	plt::legend();
	plt::title("timetobuild-recall for 1-NN");
	plt::save(prefix + "buildtime-recall.png");
}
