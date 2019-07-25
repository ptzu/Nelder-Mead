#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

struct NMpoint{
    Eigen::VectorXd vec;
    double score;
};

double cost_function(const double* x){
    return pow((100 - x[0]*x[1]), 2);
}

bool cmp(const NMpoint& a, const NMpoint& b){
    return a.score < b.score;
}

class Nelder_Mead{
public:
    Nelder_Mead(double step=0.1, int max_iter=50, double alpha=1, double gamma=2, double rho=-0.5, double sigma=0.5): 
        step(step), max_iter(max_iter), alpha(alpha), gamma(gamma), rho(rho), sigma(sigma){}

    void solve(double* para, int dim){
        // Make polytope
        std::vector<NMpoint> polytope;
        for(int i = 0; i < dim + 1; i++){
            Eigen::VectorXd vec = Eigen::Map<Eigen::VectorXd>(para, dim);
            double score;
            if(i == dim){
                score = cost_function(vec.data());
                polytope.push_back(NMpoint{vec, score});
                continue;
            }
            vec(i) += step;
            score = cost_function(vec.data());
            polytope.push_back(NMpoint{vec, score});
        }        
        int iter = 0;
        while(1){
            if(iter >= max_iter) break;
            iter++;
            sort(polytope.begin(), polytope.end(), cmp);
            // Centroid
            Eigen::VectorXd x0(dim);
            x0.setZero();
            for(int i = 0; i < polytope.size() - 1; i++){
                for(int j = 0; j < polytope[i].vec.size(); j++){
                    x0[j] += polytope[i].vec(j);
                }
            }
            x0 /= (polytope.size() - 1);
            double current_best_score = polytope[0].score;
            double worst_score = polytope.back().score;
            double second_worst_score = polytope[polytope.size() - 2].score;

            // Reflection
            Eigen::VectorXd xr = x0 + alpha * (x0 - polytope.back().vec);
            double rscore = cost_function(xr.data());
            if(current_best_score <= rscore && rscore < second_worst_score){
                polytope.pop_back();
                polytope.push_back(NMpoint{xr, rscore});
                continue;
            }

            // Expansion
            if(rscore < current_best_score){
                Eigen::VectorXd xe = x0 + gamma * (x0 - polytope.back().vec);
                double escore = cost_function(xe.data());
                if(escore < rscore){
                    polytope.pop_back();
                    polytope.push_back(NMpoint{xe, escore});
                }
                else{
                    polytope.pop_back();
                    polytope.push_back(NMpoint{xr, rscore});
                }
                continue;
            }

            // Contraction
            Eigen::VectorXd xc = x0 + rho * (x0 - polytope.back().vec);
            double cscore = cost_function(xc.data());
            if(cscore < worst_score){
                polytope.pop_back();
                polytope.push_back(NMpoint{xc, cscore});
                continue;
            }

            // Shrink
            Eigen::VectorXd x1 = polytope[0].vec;
            for(int i = 1; i < polytope.size(); i++){
                polytope[i].vec = x1 + sigma * (polytope[i].vec - x1);
            }
        }
        for(int i = 0; i < dim; i++){
            para[i] = polytope[0].vec(i);
        }
    }
private:
    const double alpha, gamma, rho, sigma;
    const double step;
    const int max_iter;
};
