#include <iostream>
#include <thread>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <sys/time.h>
# include <omp.h>
# include <openacc.h>

using namespace tbb;

static const long MATRIX_SIZE = 800;
static const long N_EXECUTIONS = 2;
static const int THREADS_NUMBER = 4;

struct Matrix {
    float **elements;

    void initialize_zero() {
        elements = new float *[MATRIX_SIZE];
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            elements[i] = new float[MATRIX_SIZE];
            for (int j = 0; j < MATRIX_SIZE; ++j) {
                elements[i][j] = 0.0f;
            }
        }
    }

    void initialize_random() {
        elements = new float *[MATRIX_SIZE];
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            elements[i] = new float[MATRIX_SIZE];
            for (int j = 0; j < MATRIX_SIZE; ++j) {
                elements[i][j] = rand() % MATRIX_SIZE;
            }
        }
    }

    void print() {
        std::cout << std::endl;
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            std::cout << "|\t";

            for (int j = 0; j < MATRIX_SIZE; ++j) {
                std::cout << elements[i][j] << "\t";
            }
            std::cout << "|" << std::endl;
        }
    }

};

Matrix m1, m2, r;

class MultiplyTBB {
public:
    void operator()(blocked_range<int> n) const {
        for (int i = n.begin(); i != n.end(); ++i) {
            for (int j = 0; j < MATRIX_SIZE; ++j) {
                float result = 0.0f;
                for (int k = 0; k < MATRIX_SIZE; ++k) {
                    const float e1 = m1.elements[i][k];
                    const float e2 = m2.elements[k][j];
                    result += e1 * e2;
                }
                r.elements[i][j] = result;
            }
        }
    }
};

void single_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2);

void benchmark_execution(
        void(*execution_function)(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2));

void multithreading_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2);

void multiply_threading(Matrix &result, const int thread_number, const Matrix &m1, const Matrix &m2);

void openmp_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2);

void openmp_optimized_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2);

void openacc_tile_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2);

void openacc_gv_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2);

void openacc_collapse_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2);

void openacc_auto_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2);

void openacc_independent_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2);

void openacc_independent_red_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2);

void tbb_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2);

int main() {
    std::cout << "Single execution" << std::endl;
    benchmark_execution(single_execution);
    std::cout << "Multi thread execution" << std::endl;
    benchmark_execution(multithreading_execution);
    std::cout << "OpenMP Heap Memory execution" << std::endl;
    benchmark_execution(openmp_execution);
    std::cout << "OpenMP Stack execution" << std::endl;
    benchmark_execution(openmp_optimized_execution);
    std::cout << "OpenACC Tile execution" << std::endl;
    benchmark_execution(openacc_tile_execution);
    std::cout << "OpenACC GV execution" << std::endl;
    benchmark_execution(openacc_gv_execution);
    std::cout << "OpenACC Collapse execution" << std::endl;
    benchmark_execution(openacc_collapse_execution);
    std::cout << "OpenACC Auto execution" << std::endl;
    benchmark_execution(openacc_auto_execution);
    std::cout << "OpenACC Independent execution" << std::endl;
    benchmark_execution(openacc_independent_execution);
    std::cout << "OpenACC Independent Reduction execution" << std::endl;
    benchmark_execution(openacc_independent_red_execution);
    std::cout << "TBB execution" << std::endl;
    benchmark_execution(tbb_execution);
    return 0;
}

void benchmark_execution(void(*execution_function)(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2)) {
    unsigned long long total_time = 0.0;
    for (int i = 0; i < N_EXECUTIONS; ++i) {
        unsigned long long elapsed_time = 0.0;
        m1.initialize_random();
        //m1.print();
        m2.initialize_random();
        //m2.print();
        r.initialize_zero();
        execution_function(r, elapsed_time, m1, m2);
        total_time += elapsed_time;
        //r.print();
    }
    std::cout << "\tAverage execution took\t" << (double) total_time / N_EXECUTIONS << " s" << std::endl;
    std::cout << "\tTotal execution took\t" << total_time << " s" << std::endl;

}

void single_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2) {
    struct timeval start, end;

    // start timer.
    gettimeofday(&start, NULL);
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            float result = 0.0f;
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                const float e1 = m1.elements[i][k];
                const float e2 = m2.elements[k][j];
                result += e1 * e2;
            }
            r.elements[i][j] = result;
        }
    }
    gettimeofday(&end, NULL);
    elapsed_time = (end.tv_sec - start.tv_sec) * 1e6;
    elapsed_time = (elapsed_time + (end.tv_usec -
                                    start.tv_usec)) * 1e-6;
}

void multiply_threading(Matrix &result, const int thread_number, const Matrix &m1, const Matrix &m2) {
    // Calculate workload
    const int n_elements = (MATRIX_SIZE * MATRIX_SIZE);
    const int n_operations = n_elements / THREADS_NUMBER;
    const int rest_operations = n_elements % THREADS_NUMBER;

    int start_op, end_op;

    if (thread_number == 0) {
        // First thread does more job
        start_op = n_operations * thread_number;
        end_op = (n_operations * (thread_number + 1)) + rest_operations;
    } else {
        start_op = n_operations * thread_number + rest_operations;
        end_op = (n_operations * (thread_number + 1)) + rest_operations;
    }

    for (int op = start_op; op < end_op; ++op) {
        const int row = op % MATRIX_SIZE;
        const int col = op / MATRIX_SIZE;
        float r = 0.0f;
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            const float e1 = m1.elements[row][i];
            const float e2 = m2.elements[i][col];
            r += e1 * e2;
        }

        result.elements[row][col] = r;
    }
}

void multithreading_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2) {
    struct timeval start, end;

    // start timer.
    gettimeofday(&start, NULL);
    std::thread threads[THREADS_NUMBER];

    for (int i = 0; i < THREADS_NUMBER; ++i) {
        threads[i] = std::thread(multiply_threading, std::ref(r), i, std::ref(m1), std::ref(m2));
    }

    for (int i = 0; i < THREADS_NUMBER; ++i) {
        threads[i].join();
    }

    gettimeofday(&end, NULL);
    elapsed_time = (end.tv_sec - start.tv_sec) * 1e6;
    elapsed_time = (elapsed_time + (end.tv_usec -
                                    start.tv_usec)) * 1e-6;
}

void openmp_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2) {
    struct timeval start, end;

    // start timer.
    gettimeofday(&start, NULL);
#pragma omp parallel for
    for(int i=0; i<MATRIX_SIZE; i++){
        for(int j=0; j<MATRIX_SIZE; j++){
            for(int k=0; k<MATRIX_SIZE; k++){
                r.elements[i][j] += m1.elements[i][k] * m2.elements[k][j];
            }
        }
    }
    gettimeofday(&end, NULL);
    elapsed_time = (end.tv_sec - start.tv_sec) * 1e6;
    elapsed_time = (elapsed_time + (end.tv_usec -
                                    start.tv_usec)) * 1e-6;
}

void openmp_optimized_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2) {
    struct timeval start, end;
    int i,j,k;

    // start timer.
    gettimeofday(&start, NULL);
#pragma omp parallel shared(m1, m2, r) private(i, j, k) num_threads(THREADS_NUMBER)
    {
#pragma omp for  schedule(static)
        for (i = 0; i < MATRIX_SIZE; ++i) {
            for (j = 0; j < MATRIX_SIZE; ++j) {
                float result = 0.0f;
                for (k = 0; k < MATRIX_SIZE; ++k) {
                    const float e1 = m1.elements[i][k];
                    const float e2 = m2.elements[k][j];
                    result += e1 * e2;
                }
                r.elements[i][j] = result;
            }
        }
    }
    gettimeofday(&end, NULL);
    elapsed_time = (end.tv_sec - start.tv_sec) * 1e6;
    elapsed_time = (elapsed_time + (end.tv_usec -
                                    start.tv_usec)) * 1e-6;
}

void openacc_tile_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2) {
    struct timeval start, end;

    // start timer.
    gettimeofday(&start, NULL);
#pragma acc data copyin(m1, m2) copy(r)
#pragma acc kernels
#pragma acc loop tile(32, 32)
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        //todo multiple options here
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            float result = 0.0f;
#pragma acc loop reduction(+:result)
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                const float e1 = m1.elements[i][k];
                const float e2 = m2.elements[k][j];
                result += e1 * e2;
            }
            r.elements[i][j] = result;
        }
    }
    gettimeofday(&end, NULL);
    elapsed_time = (end.tv_sec - start.tv_sec) * 1e6;
    elapsed_time = (elapsed_time + (end.tv_usec -
                                    start.tv_usec)) * 1e-6;
}

void openacc_gv_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2) {
    struct timeval start, end;

    // start timer.
    gettimeofday(&start, NULL);
#pragma acc data copyin(m1, m2) copy(r)
#pragma acc kernels
#pragma acc loop gang(32)
    for (int i = 0; i < MATRIX_SIZE; ++i) {
#pragma acc loop vector(16)
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            float result = 0.0f;
#pragma acc loop reduction(+:result)
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                const float e1 = m1.elements[i][k];
                const float e2 = m2.elements[k][j];
                result += e1 * e2;
            }
            r.elements[i][j] = result;
        }
    }

    gettimeofday(&end, NULL);
    elapsed_time = (end.tv_sec - start.tv_sec) * 1e6;
    elapsed_time = (elapsed_time + (end.tv_usec -
                                    start.tv_usec)) * 1e-6;
}

void openacc_collapse_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2) {
    struct timeval start, end;

    // start timer.
    gettimeofday(&start, NULL);
#pragma acc data copyin(m1, m2) copy(r)
#pragma acc kernels
#pragma acc loop collapse(2) independent
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            float result = 0.0f;
#pragma acc loop reduction(+:result)
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                const float e1 = m1.elements[i][k];
                const float e2 = m2.elements[k][j];
                result += e1 * e2;
            }
            r.elements[i][j] = result;
        }
    }
    gettimeofday(&end, NULL);
    elapsed_time = (end.tv_sec - start.tv_sec) * 1e6;
    elapsed_time = (elapsed_time + (end.tv_usec -
                                    start.tv_usec)) * 1e-6;
}

void openacc_auto_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2) {
    struct timeval start, end;

    // start timer.
    gettimeofday(&start, NULL);
#pragma acc data copyin(m1, m2) copy(r)
#pragma acc kernels
#pragma acc loop auto
    for (int i = 0; i < MATRIX_SIZE; ++i) {
#pragma acc loop auto
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            float result = 0.0f;
#pragma acc loop auto
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                const float e1 = m1.elements[i][k];
                const float e2 = m2.elements[k][j];
                result += e1 * e2;
            }
            r.elements[i][j] = result;
        }
    }

    gettimeofday(&end, NULL);
    elapsed_time = (end.tv_sec - start.tv_sec) * 1e6;
    elapsed_time = (elapsed_time + (end.tv_usec -
                                    start.tv_usec)) * 1e-6;
}

void openacc_independent_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2) {
    struct timeval start, end;

    // start timer.
    gettimeofday(&start, NULL);
#pragma acc data copyin(m1, m2) copy(r)
#pragma acc kernels
#pragma acc loop independent
    for (int i = 0; i < MATRIX_SIZE; ++i) {
#pragma acc loop independent
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            float result = 0.0f;
#pragma acc loop seq
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                const float e1 = m1.elements[i][k];
                const float e2 = m2.elements[k][j];
                result += e1 * e2;
            }
            r.elements[i][j] = result;
        }
    }

    gettimeofday(&end, NULL);
    elapsed_time = (end.tv_sec - start.tv_sec) * 1e6;
    elapsed_time = (elapsed_time + (end.tv_usec -
                                    start.tv_usec)) * 1e-6;
}

void
openacc_independent_red_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2) {
    struct timeval start, end;

    // start timer.
    gettimeofday(&start, NULL);
#pragma acc data copyin(m1, m2) copy(r)
#pragma acc kernels
#pragma acc loop independent
    for (int i = 0; i < MATRIX_SIZE; ++i) {
#pragma acc loop independent
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            float result = 0.0f;
#pragma acc loop reduction(+:result)
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                const float e1 = m1.elements[i][k];
                const float e2 = m2.elements[k][j];
                result += e1 * e2;
            }
            r.elements[i][j] = result;
        }
    }

    gettimeofday(&end, NULL);
    elapsed_time = (end.tv_sec - start.tv_sec) * 1e6;
    elapsed_time = (elapsed_time + (end.tv_usec -
                                    start.tv_usec)) * 1e-6;
}

void tbb_execution(Matrix &r, unsigned long long &elapsed_time, const Matrix &m1, const Matrix &m2) {
    struct timeval start, end;

    // start timer.
    gettimeofday(&start, NULL);
    parallel_for(blocked_range<int>(0, MATRIX_SIZE), MultiplyTBB());

    gettimeofday(&end, NULL);
    elapsed_time = (end.tv_sec - start.tv_sec) * 1e6;
    elapsed_time = (elapsed_time + (end.tv_usec -
                                    start.tv_usec)) * 1e-6;
}