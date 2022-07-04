#include <torch/script.h>
#include <cmath>
#include <iostream>
#include <range/v3/view.hpp>
#include "cnpy.h"

#define iterations 200000
#define recall_cost 3.0f
#define forget_cost 9.0f
#define target_halflife 365
#define hidden_dim 2
#define inf 1000
#define epsilon 0.01f
#define state_limit (int) (2.0f/epsilon)

using namespace std;
using namespace ranges::view;

torch::jit::script::Module module;
vector<torch::jit::IValue> inputs;


array<float, hidden_dim> find_init_state() {
    torch::Tensor line_tensor = torch::empty({1, 1, 3});
    torch::Tensor hidden_tensor = torch::empty({1, 1, hidden_dim});

    auto line_accessor = line_tensor.accessor<float, 3>();
    auto hidden_accessor = hidden_tensor.accessor<float, 3>();

    line_accessor[0][0][0] = 0.0f;
    line_accessor[0][0][1] = 0.0f;
    line_accessor[0][0][2] = 0.34f;
    for (int i = 0; i < hidden_dim; i++) {
        hidden_accessor[0][0][i] = 0.0f;
    }

    inputs.emplace_back(line_tensor);
    inputs.emplace_back(hidden_tensor);
    auto output = module.forward(inputs).toTuple();
    cout << output->elements()[0].toTensor()[0][0].item<float>() << endl;
    array<float, hidden_dim> init_state{};
    for (int i = 0; i < hidden_dim; i++) {
        cout << output->elements()[1].toTensor()[0][0][i].item<float>() << endl;
        hidden_accessor[0][0][i] = output->elements()[1].toTensor()[0][0][i].item<float>();
        init_state[i] = hidden_accessor[0][0][i];
    }
    inputs.clear();
    output.reset();

    auto halflife = module.run_method("full_connect", hidden_tensor).toTensor()[0][0].item<float>();
    halflife = exp(halflife);
    cout << halflife << endl;
    return init_state;
}

int load_model() {
    try {
        module = torch::jit::load("../tmp/nn-GRU_nh-2_loss-sMAPE/model.pt");
    }
    catch (const c10::Error &e) {
        cerr << "error loading the model\n";
        return -1;
    }
    cout << "model load ok\n";
    module.eval();
    return 0;
}

float state2halflife(array<float, hidden_dim> x) {
    torch::Tensor hidden_tensor = torch::empty({1, 1, hidden_dim});
    auto hidden_accessor = hidden_tensor.accessor<float, 3>();
    for (int i = 0; i < hidden_dim; i++) {
        hidden_accessor[0][0][i] = x[i];
    }
    auto halflife = module.run_method("full_connect", hidden_tensor).toTensor()[0][0].item<float>();
    halflife = exp(halflife);
    return halflife;
}

array<float, hidden_dim> cal_next_recall_state(array<float, hidden_dim> x, int response, int interval, float recall) {
    torch::Tensor line_tensor = torch::empty({1, 1, 3});
    torch::Tensor hidden_tensor = torch::empty({1, 1, hidden_dim});

    auto line_accessor = line_tensor.accessor<float, 3>();
    auto hidden_accessor = hidden_tensor.accessor<float, 3>();

    line_accessor[0][0][0] = (float) response;
    line_accessor[0][0][1] = (float) interval;
    line_accessor[0][0][2] = recall;
    for (int i = 0; i < hidden_dim; i++) {
        hidden_accessor[0][0][i] = x[i];
    }

    inputs.emplace_back(line_tensor);
    inputs.emplace_back(hidden_tensor);
    auto output = module.forward(inputs).toTuple();
    inputs.clear();

    array<float, hidden_dim> new_state{};
    for (int i = 0; i < hidden_dim; i++) {
        new_state[i] = output->elements()[1].toTensor()[0][0][i].item<float>();
    }
    return new_state;
}

int state2index(float s) {
    return max(min((int) ((s + 1.0f) / epsilon), state_limit - 1), 0);
}

array<int, hidden_dim> states2indexes(array<float, hidden_dim> s) {
    array<int, hidden_dim> indexes{};
    for (int i = 0; i < hidden_dim; i++) {
        indexes[i] = state2index(s[i]);
    }
    return indexes;
}

int main() {
    load_model();
    auto init_state = find_init_state();

//    array<
//    array<
    array<
    array<float, state_limit>
    , state_limit>
//    , state_limit>
//    , state_limit>
    cost_list{};

//    array<
//    array<
    array<
    array<float, state_limit>
    , state_limit>
//    , state_limit>
//    , state_limit>
    recall_list{};

//    array<
//    array<
    array<
    array<int, state_limit>
    , state_limit>
//    , state_limit>
//    , state_limit>
    used_interval_list{};
//    array<
//    array<
    array<
    array<float, state_limit>
    , state_limit>
//    , state_limit>
//    , state_limit>
    halflife_list{};

    for (int x1 = 0; x1 < state_limit; x1++) {
        for (int x2 = 0; x2 < state_limit; x2++) {
//            for (int x3 = 0; x3 < state_limit; x3++) {
//                for (int x4 = 0; x4 < state_limit; x4++) {
                    array<float, hidden_dim> state = {(float) x1 * epsilon - 1.0f + epsilon / 2.0f
                                                      ,(float) x2 * epsilon - 1.0f + epsilon / 2.0f
//                                                      ,(float) x3 * epsilon - 1.0f + epsilon / 2.0f
//                                                      ,(float) x4 * epsilon - 1.0f + epsilon / 2.0f
                    };
                    auto halflife = state2halflife(state);
                    used_interval_list
                    [x1]
                    [x2]
//                    [x3]
//                    [x4]
                    = 0;
                    halflife_list
                    [x1]
                    [x2]
//                    [x3]
//                    [x4]
                    = halflife;
                    if (halflife <= target_halflife) {
                        cost_list
                        [x1]
                        [x2]
//                        [x3]
//                        [x4]
                        = inf;
                    } else {
                        cost_list
                        [x1]
                        [x2]
//                        [x3]
//                        [x4]
                        = 0.0f;
                        recall_list
                        [x1]
                        [x2]
//                        [x3]
//                        [x4]
                        = 0.0f;
                    }
                }
            }
//        }
//    }
    vector<float> raw_halflife;
    auto half_life =
//            join(
//            join(
            join(
            halflife_list
//            )
//            )
            );
    raw_halflife.assign(half_life.begin(), half_life.end());
    cnpy::npy_save("./gru_half_life.npy", &raw_halflife[0],
                   {state_limit
                    , state_limit
//                    , state_limit
//                    , state_limit
                    }, "w");

    cout << "matrix load_model ok\n";
    int start_time = (int) time((time_t *) nullptr);
    for (int i = 0; i < iterations; ++i) {
        float h0_cost = cost_list[state2index(init_state[0])]
        [state2index(init_state[1])]
//        [state2index(init_state[2])]
//        [state2index(init_state[3])]
        ;
        for (int x1 = 0; x1 < state_limit; x1++) {
            for (int x2 = 0; x2 < state_limit; x2++) {
//                for (int x3 = 0; x3 < state_limit; x3++) {
//                    for (int x4 = 0; x4 < state_limit; x4++) {
                        array<float, hidden_dim> state = {(float) x1 * epsilon - 1.0f + epsilon / 2.0f
                                                         ,(float) x2 * epsilon - 1.0f + epsilon / 2.0f
//                                                      ,(float) x3 * epsilon - 1.0f + epsilon / 2.0f
//                                                      ,(float) x4 * epsilon - 1.0f + epsilon / 2.0f
                        };

                        float halflife = state2halflife(state);
                        if (halflife >= target_halflife) continue;

                        int interval_min = 1;
                        int interval_max = max(1, (int) round(halflife));
                        int interval_step = max(1, (int) round((interval_max - interval_min) / 20));


                        for (int interval = interval_max;
                             interval >= interval_min; interval = interval - interval_step) {

                            float p_recall = exp2(-(float) interval / halflife);

                            auto recall_state = cal_next_recall_state(state, 1, interval, p_recall);
                            auto forget_state = cal_next_recall_state(state, 0, interval, p_recall);
                            auto recall_index = states2indexes(recall_state);
                            auto forget_index = states2indexes(forget_state);

                            float exp_cost =
                                    p_recall *
                                    (cost_list
                                    [recall_index[0]]
                                    [recall_index[1]]
//                                    [recall_index[2]]
//                                    [recall_index[3]]
                                    + recall_cost) +
                                    (1.0f - p_recall) *
                                    (cost_list
                                    [forget_index[0]]
                                    [forget_index[1]]
//                                    [forget_index[2]]
//                                    [forget_index[3]]
                                    + forget_cost);

                            if (exp_cost < cost_list
                            [x1]
                            [x2]
//                            [x3]
//                            [x4]
                            ) {
                                cost_list
                                [x1]
                                [x2]
//                                [x3]
//                                [x4]
                                = exp_cost;
                                used_interval_list
                               [x1]
                               [x2]
//                               [x3]
//                               [x4]
                               = interval;

                                recall_list
                                [x1]
                                [x2]
//                               [x3]
//                               [x4]
                               = p_recall;
                            }
                        }
                    }
                }
//            }
//        }

        float diff = h0_cost -
                     cost_list
                     [state2index(init_state[0])]
                     [state2index(init_state[1])]
//                     [state2index(init_state[2])]
//                     [state2index(init_state[3])]
                     ;
        printf("iter %d\tdiff %f\ttime %ds\tcost %f\n", i, diff,
               (int) time((time_t *) nullptr) - start_time,
               cost_list
               [state2index(init_state[0])]
               [state2index(init_state[1])]
//               [state2index(init_state[2])]
//               [state2index(init_state[3])]
               );
        if (i % 5 == 0) {
            vector<float> raw_cost;
            auto cost =
//                    join(
//                    join(
                    join(
                    cost_list
//                    )
//                    )
                    );
            printf("cost accumulate %f\n", accumulate(cost.begin(), cost.end(), 0.0f));
            raw_cost.assign(cost.begin(), cost.end());
            cnpy::npy_save("./gru_cost.npy", &raw_cost[0],
                           {state_limit
                            , state_limit
//                            , state_limit
//                            , state_limit
                            }, "w");
            vector<float> raw_recall;
            auto recall =
//                    join(
//                    join(
                    join(
                    recall_list
//                    )
//                    )
                    );
            raw_recall.assign(recall.begin(), recall.end());
            cnpy::npy_save("./gru_recall.npy", &raw_recall[0],
                           {state_limit
                            , state_limit
//                            , state_limit
//                            , state_limit
                            }, "w");
            vector<int> raw_policy;
            auto policy =
//                    join(
//                    join(
                    join(
                    used_interval_list
//                    )
//                    )
                    );
            printf("policy accumulate %d\n", accumulate(policy.begin(), policy.end(), 0));
            raw_policy.assign(policy.begin(), policy.end());
            cnpy::npy_save("./gru_policy.npy", &raw_policy[0],
                           {state_limit
                            , state_limit
//                            , state_limit
//                            , state_limit
                            }, "w");

        }
        if (diff < 1 && h0_cost < inf) {
            vector<float> raw_cost;
            auto cost =
//                    join(
//                    join(
                    join(
                    cost_list
//                    )
//                    )
                    );
            raw_cost.assign(cost.begin(), cost.end());
            cnpy::npy_save("./gru_cost.npy", &raw_cost[0],
                           {state_limit
                            , state_limit
//                            , state_limit
//                            , state_limit
                            }, "w");
            vector<float> raw_recall;
            auto recall =
//                    join(
//                    join(
                    join(
                            recall_list
//                    )
//                    )
                    );
            raw_recall.assign(recall.begin(), recall.end());
            cnpy::npy_save("./gru_recall.npy", &raw_recall[0],
                           {state_limit
                            , state_limit
//                            , state_limit
//                            , state_limit
                            }, "w");
            vector<int> raw_policy;
            auto policy =
//                    join(
//                    join(
                    join(
                    used_interval_list
//                    )
//                    )
                    );
            raw_policy.assign(policy.begin(), policy.end());
            cnpy::npy_save("./gru_policy.npy", &raw_policy[0],
                           {state_limit
                            , state_limit
//                            , state_limit
//                            , state_limit
                            }, "w");
            break;
        }
    }

    return 0;
}