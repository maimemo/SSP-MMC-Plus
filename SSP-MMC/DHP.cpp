#include <cmath>
#include <cstdio>
#include <ctime>
#include <array>
#include <fstream>
#include <range/v3/view.hpp>
#include "cnpy.h"

#define iterations 200000
#define recall_cost 3.0f
#define forget_cost 9.0f
#define inf 1000
#define max_index 122
#define min_index -40
#define base 1.05
#define d_limit 18
#define d_offset 2

using namespace std;
using namespace ranges::view;

float cal_start_halflife(int difficulty) {
    float p = max(0.925 - 0.05 * difficulty, 0.025);
    return -1 / log2(p);
}

float cal_next_recall_halflife(float h, float p, int d, int recall) {
    if (recall == 1) {
        return h * (1 + exp(3.252) * pow(d, -0.3855) * pow(h, -0.1471) * pow(1 - p, 0.8214));
    } else {
        return exp(1.003) * pow(d, -0.1524) * pow(h, 0.2648) * pow(1 - p, -0.01736);
    }
}

int cal_halflife_index(float h) {
    return (int) round(log(h) / log(base)) - min_index;
}

float cal_index_halflife(int index) {
    return exp((index + min_index) * log(base));
}

int main() {
    array<float, max_index - min_index> halflife_list{};
    for (int i = 0; i < max_index - min_index; i++) {
        halflife_list[i] = pow(base, i + min_index);
    }
    int index_len = max_index - min_index;
    array<array<float, max_index - min_index>, d_limit> cost_list{};
    for (int d = 1; d <= d_limit; d++) {
        for (int i = 0; i < index_len - 1; i++) {
            cost_list[d - 1][i] = (float) inf;
        }
        cost_list[d - 1][index_len - 1] = 0;
    }
    array<array<int, max_index - min_index>, d_limit> used_interval_list{};
    array<array<float, max_index - min_index>, d_limit> recall_list{};
    array<array<int, max_index - min_index>, d_limit> next_index{};
    int start_time = (int) time((time_t *) nullptr);
    for (int d = d_limit; d >= 1; d--) {
        float h0 = cal_start_halflife(d);
        int h0_index = cal_halflife_index(h0);
        for (int i = 0; i < iterations; ++i) {
            float h0_cost = cost_list[d - 1][h0_index];
            for (int h_index = index_len - 2; h_index >= 0; h_index--) {
                float halflife = halflife_list[h_index];

                int interval_min;
                int interval_max;

                interval_min = 1;
                interval_max = max(1, (int) round(halflife * log(0.3) / log(0.5)));

                for (int interval = interval_max; interval >= interval_min; interval--) {
                    float p_recall = exp2(-interval / halflife);
                    float recall_h = cal_next_recall_halflife(halflife, p_recall, d, 1);
                    float forget_h = cal_next_recall_halflife(halflife, p_recall, d, 0);
                    int recall_h_index = min(cal_halflife_index(recall_h), index_len - 1);
                    int forget_h_index = max(cal_halflife_index(forget_h), 0);
                    float exp_cost =
                            p_recall * (cost_list[d - 1][recall_h_index] + recall_cost) +
                            (1.0 - p_recall) *
                            (cost_list[min(d - 1 + d_offset, d_limit - 1)][forget_h_index] + forget_cost);
                    if (exp_cost < cost_list[d - 1][h_index]) {
                        cost_list[d - 1][h_index] = exp_cost;
                        used_interval_list[d - 1][h_index] = interval;
                        recall_list[d - 1][h_index] = p_recall;
                        next_index[d - 1][h_index] = recall_h_index;
                    }
                }
            }

            float diff = h0_cost - cost_list[d - 1][h0_index];
            printf("D %d\titer %d\tdiff %f\ttime %ds\tcost %f\n", d, i, diff,
                   (int) time((time_t *) nullptr) - start_time,
                   cost_list[d - 1][h0_index]);
            if (diff < 1 && h0_cost < inf) {
                vector<float> raw_recall;
                auto recall = join(recall_list);
                raw_recall.assign(recall.begin(), recall.end());
                cnpy::npy_save("./dhp_recall.npy", &raw_recall[0],
                               {d_limit, max_index - min_index}, "w");
                vector<float> raw_cost;
                auto cost = join(cost_list);
                raw_cost.assign(cost.begin(), cost.end());
                cnpy::npy_save("./dhp_cost.npy", &raw_cost[0],
                               {d_limit, max_index - min_index}, "w");
                vector<int> raw_policy;
                auto policy = join(used_interval_list);
                raw_policy.assign(policy.begin(), policy.end());
                cnpy::npy_save("./dhp_policy.npy", &raw_policy[0],
                               {d_limit, max_index - min_index}, "w");
                break;
            }
        }
    }
    return 0;
}