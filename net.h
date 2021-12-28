#ifndef  _NET__H
#define  _NET__H
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <new>

#include "function.h"

using u32 = unsigned long;

const  u32  max_layernum = 10;

enum class train_type {
    Grad=1,
    Momentum,
    RMSprop,
    Adam
};

struct node {
    double value{}, bias{}, bias_delta{}, factor{};
    std::vector<double> weight, weight_delta;

    double v_bias_delta{}, s_bias_delta{};
    std::vector<double> v_weight_delta, s_weight_delta;
};

struct traindata {
    std::vector<double> in, out;
};

class  net {

private:
    /*-----------超参数-----------*/
    double alpha = 0.2;                                         //学习率
    double beta1 = 0, beta2 = 0, epsilon = 0;                   //adma算法参数
    double d_beta1=1, s_beta2=1;                                    //作偏差修正用
    double lambada = 0;                                         //正则化参数
    u32 mini_batch_size = 0;                                    //mini_batch
    u32 layernum = 4;                                           //层数
    std::vector<u32>  layernode;                                //每层节点个数
    /*-----------参数-------------*/
    double lost = 0, cost = 0;                                  //损失函数，成本函数
    /*-----------网络-------------*/
    std::vector< node* > layer[max_layernum];
    /*-----------训练集&&测试集-------------*/
    std::vector<traindata> trainset, testset;
    /*-----------日志-------------*/
    std::vector<double>      costsum;                           //成本序列，绘图用
    std::vector<std::string> logo;                              //训练日志

    bool init_tage = false;                                     //初始化标签

public:
    net() :alpha(0.1), beta1(0.9), beta2(0.999), epsilon(1e-8) {};
    net(double alpha) :alpha(alpha) {};
    net(double alpha, double bata1, double bata2, double epsilon) :alpha(alpha), beta1(bata1), beta2(beta2), epsilon(epsilon) {};

    ~net() {
        destroy();
    };

    void set_alpha(double res) { alpha = res; return; }
    void set_beta1(double res) { beta1 = res; return; }
    void set_beta2(double res) { beta2 = res; return; }
    void set_epsilon(double res) { epsilon = res; return; }
    void set_lambada(double res) { lambada = res; return; }
    void set_layernum(u32 res) { layernum = res; return; }
    void set_layernode(std::vector<u32> res) { layernode = res; return; }
    void set_mini_batch_size(double res) { mini_batch_size = res; return; }
    void reset_delta(void);
    void train_normalization(void);
    void test_normalization(void);

    bool load(std::string filename);
    bool load_trainset(std::string filename);
    bool load_testset(std::string filename);
    bool download(std::string filename);
    bool init();
    bool train(int num,train_type type);
    bool train(double maxcost);
    bool prediction(std::string filename);

    void destroy(void);

};

void net::reset_delta(void) {
    for (int i = 0; i < layernum; ++i) {
        for (int j = 0; j < layernode[i]; ++j) {
            layer[i][j]->bias_delta = 0.f;
            layer[i][j]->weight_delta.assign(layer[i][j]->weight_delta.size(), 0.f);
        }
    }
    return;
}
void net::train_normalization(void) {
    /*-------------均值归一化-------------*/
    double mean = 0;
    for (auto &a : trainset) {
        for (auto &b : a.in)
            mean += b;
    }
    mean /= trainset.size() * (trainset[0].in.size());
    for (auto &a : trainset) {
        for (auto &b : a.in)
            b -= mean;
    }
    /*-------------方差归一化-------------*/
    double var = 0;
    for (auto &a : trainset) {
        for (auto &b : a.in)
            var += b*b;
    }
    var /= trainset.size() * (trainset[0].in.size());
    for (auto &a : trainset) {
        for (auto &b : a.in)
            b /= var;
    }
    return;
}
void net::test_normalization(void) {

    return;
}
bool net::load(std::string filename) {
    return true;
}
bool net::load_trainset(std::string filename) {
    std::ifstream fin(filename);
    double temp;
    int n, m1, m2;
    fin >> m1 >> m2 >> n;
    if (n <= 0)
        return false;
    for (int i = 0; i < n; ++i) {
        traindata res;
        for (int j = 0; j < m1; ++j) {
            fin >> temp;
            res.in.push_back(temp);
        }
        for (int j = 0; j < m2; ++j) {
            fin >> temp;
            res.out.push_back(temp);
        }
        trainset.push_back(res);
    }
    return true;
}
bool net::load_testset(std::string filename) {
    std::ifstream fin(filename);
    double temp;
    int n, m;
    fin >> m >> n;
    if (n <= 0)
        return false;
    for (int i = 0; i < n; ++i) {
        traindata res;
        for (int j = 0; j < m; ++j) {
            fin >> temp;
            res.in.push_back(temp);
        }
        testset.push_back(res);
    }
    return true;
}
bool net::download(std::string filename) {
    std::ofstream os;
    os.open(filename);
    os << std::setprecision(9) << std::endl;
    for (auto a : costsum)
        os << a << std::endl;
    os.close();
    return true;
}
bool net::init() {
    std::mt19937 rd;
    rd.seed(3);
    std::uniform_real_distribution<double> dis(-0.5, 0.5);

    for (int i = 0; i < layernum; ++i) {
        for (int j = 0; j < layernode[i]; ++j) {
            node* ptr = new node;
            layer[i].push_back(ptr);
            layer[i][j]->bias = dis(rd);
            layer[i][j]->s_bias_delta = 0.0;
            layer[i][j]->v_bias_delta = 0.0;
            if (i + 1 == layernum)
                continue;
            double res;
            if (i + 2 == layernum)
                res = sqrt(1.0 / layernode[i]);
            else
                res = sqrt(2.0 / layernode[i]);

            for (int k = 0; k < layernode[i + 1]; ++k) {
                layer[i][j]->weight.push_back(dis(rd) * res);
                layer[i][j]->weight_delta.push_back(0.0);
                layer[i][j]->v_weight_delta.push_back(0.0);
                layer[i][j]->s_weight_delta.push_back(0.0);
            }
        }
    }
    init_tage = true;
    return true;
}
bool net::train(int num,  train_type type) {

    if (num <= 0)
        return false;

    if (!init_tage)
        init();

    d_beta1 = 1;
    s_beta2 = 1;
    while (num--) {
        d_beta1 *= beta1;
        s_beta2 *= beta2;
        cost = 0.0;
        reset_delta();
        for (int cnt = 0; cnt < trainset.size(); ++cnt) {
            /*-----------------------------填充输入层-----------------------------*/
            std::vector<double> input = trainset[cnt].in;
            std::vector<double> output = trainset[cnt].out;
            for (int i = 0; i < input.size(); ++i) {
                layer[0][i]->value = input[i];
            }
            /*-----------------------------正向传播-----------------------------*/
            for (int i = 1; i < layernum; ++i) {
                for (int j = 0; j < layernode[i]; ++j) {
                    double u = 0.0;
                    for (int k = 0; k < layernode[i - 1]; ++k) {
                        u += layer[i - 1][k]->value * layer[i - 1][k]->weight[j];
                    }
                    u -= layer[i][j]->bias;
                    if (i + 1 == layernum)
                        layer[i][j]->value = sigmoid(u);
                    else
                        layer[i][j]->value = relu(u);
                }
            }
            
            if (output.size() == 1)
                lost = lostvalue(output[0], layer[layernum - 1][0]->value);
            cost += lost;

            /*-----------------------------反向传播-----------------------------*/

            //从倒数第一层开始更新，更新自身的公因子后，依次更新该层的bias和该层之前的weight，
            for (int i = layernum - 1; i > 0; --i) {

                /*-----------------------------计算因子-----------------------------*/
                if (i + 1 == layernum) {
                    if (output.size() == 1) {
                        layer[layernum - 1][0]->factor = ((1.0 - output[0]) / (1.0 - layer[layernum - 1][0]->value)
                            - (output[0]) / (layer[layernum - 1][0]->value)) * layer[layernum - 1][0]->value * (1.0 - layer[layernum - 1][0]->value);
                    }
                }
                else {
                    
                    for (int j = 0; j < layernode[i]; ++j) {
                        layer[i][j]->factor = 0.0;
                        for (int k = 0; k < layernode[i + 1]; ++k) {
                            layer[i][j]->factor += layer[i][j]->weight[k] *
                                layer[i + 1][k]->factor * d_relu(layer[i + 1][k]->value);
                        }
                    }                  
                }             
                /*-----------------------------计算梯度-----------------------------*/
                for (int j = 0; j < layernode[i]; ++j) {                   
                    layer[i][j]->bias_delta -= layer[i][j]->factor;
                    for (int k = 0; k < layernode[i - 1]; ++k) {           
                        layer[i - 1][k]->weight_delta[j] += layer[i][j]->factor * layer[i - 1][k]->value;
                    }
                }               
            }
        }

        /*-----------------------------参数更新-----------------------------*/

        for (int i = 0; i < layernum; ++i) {
            for (int j = 0; j < layernode[i]; ++j) {
                if (i != 0) {
                    double d_b= layer[i][j]->bias_delta / trainset.size();
                    switch (type) {
                    case train_type::Grad:
                        layer[i][j]->bias -= alpha * d_b;
                        break;
                    case train_type::Momentum:
                        layer[i][j]->v_bias_delta = beta1 * layer[i][j]->v_bias_delta + (1 - beta1) * d_b;
                        layer[i][j]->bias -= alpha * layer[i][j]->v_bias_delta / (1-d_beta1);
                        break;
                    case train_type::RMSprop:
                        layer[i][j]->s_bias_delta = beta2 * layer[i][j]->s_bias_delta + (1 - beta2) * d_b * d_b;
                        layer[i][j]->bias -= alpha * d_b / (sqrt(layer[i][j]->s_bias_delta / (1 - s_beta2)) + epsilon);
                        break;
                    case train_type::Adam:
                        layer[i][j]->v_bias_delta = beta1 * layer[i][j]->v_bias_delta + (1 - beta1) * d_b;
                        layer[i][j]->s_bias_delta = beta2 * layer[i][j]->s_bias_delta + (1 - beta2) * d_b * d_b;
                        layer[i][j]->bias -= alpha * (layer[i][j]->v_bias_delta / (1 - d_beta1)) / (sqrt(layer[i][j]->s_bias_delta / (1 - s_beta2)) + epsilon);
                        break;
                    }
                }
                if (i != layernum - 1) {
                    for (int k = 0; k < layer[i][j]->weight.size(); ++k) {
                        double d_w= layer[i][j]->weight_delta[k] / trainset.size();
                        switch (type) {
                        case train_type::Grad:
                            layer[i][j]->weight[k] -= alpha * d_w;
                            break;
                        case train_type::Momentum:
                            layer[i][j]->v_weight_delta[k] = beta1 * layer[i][j]->v_weight_delta[k] + (1 - beta1) * d_w;
                            layer[i][j]->weight[k] -= alpha * layer[i][j]->v_weight_delta[k] / (1 - d_beta1);
                            break;
                        case train_type::RMSprop:
                            layer[i][j]->s_weight_delta[k] = beta2 * layer[i][j]->s_weight_delta[k] + (1 - beta2) * d_w * d_w;
                            layer[i][j]->weight[k] -= alpha * d_w / (sqrt(layer[i][j]->s_weight_delta[k] / (1 - s_beta2)) + epsilon);
                            break;
                        case train_type::Adam:
                            layer[i][j]->v_weight_delta[k] = beta1 * layer[i][j]->v_weight_delta[k] + (1 - beta1) * d_w;
                            layer[i][j]->s_weight_delta[k] = beta2 * layer[i][j]->s_weight_delta[k] + (1 - beta2) * d_w * d_w;
                            layer[i][j]->weight[k] -= alpha * (layer[i][j]->v_weight_delta[k] / (1 - d_beta1)) / (sqrt(layer[i][j]->s_weight_delta[k] / (1 - s_beta2)) + epsilon);
                            break;
                        }
                    }
                }
            }
        }
        cost /= trainset.size();
        costsum.push_back(cost);
    }

    return true;
}
bool net::train(double maxcost) {
    return true;
}
bool net::prediction(std::string filename) {
    for (int cnt = 0; cnt < testset.size(); ++cnt) {
        /*-----------------------------填充输入层-----------------------------*/
        std::vector<double> input = testset[cnt].in;
        std::vector<double>* output = &(testset[cnt].out);
        for (int i = 0; i < input.size(); ++i) {
            layer[0][i]->value = input[i];
        }
        /*-----------------------------正向传播-----------------------------*/
        for (int i = 1; i < layernum; ++i) {
            for (int j = 0; j < layernode[i]; ++j) {
                double u = 0.0;
                for (int k = 0; k < layernode[i - 1]; ++k) {
                    u += layer[i - 1][k]->value * layer[i - 1][k]->weight[j];
                }
                u -= layer[i][j]->bias;
                if (i + 1 == layernum) {
                    layer[i][j]->value = sigmoid(u);
                    output->push_back(layer[i][j]->value);
                }
                else
                    layer[i][j]->value = relu(u);
            }
        }
    }
    std::ofstream os;
    os.open(filename);
    os << std::setprecision(4) << std::endl;
    for (auto a : testset) {
        for (auto b : a.in)
            os << b << "  ";
        for (auto b : a.out)
            os << b << std::endl;
    }
    os.close();
    return true;
}
void net::destroy(void) {

    for (int i = 0; i < layernum; ++i) {
        for (int j = 0; j < layernode[i]; ++j) {
            delete layer[i][j];
        }
    }
    return;
}

#endif
