
#include <CL/sycl.hpp>
#include <stdio.h>

using namespace cl::sycl;

int main(int argc, char* argv[]) {
    int length = 1024;
    //int length = 1 << 28;
    clock_t start, end;

    try {
        // cola de trabajo
        sycl::queue q(sycl::default_selector{});//GPU(si hay) o CPU
        //sycl::queue q(sycl::cpu_selector{});
        //sycl::queue q(sycl::gpu_selector{});

        // presenta información del dispositivo usado
        std::cout << "Dispositivo: " << q.get_device().get_info<info::device::name>() << std::endl;

        // usamos Unified Shared Memory, USM
        auto A = sycl::malloc_shared<int>(length, q);
        auto B = sycl::malloc_shared<int>(length, q);
        auto C = sycl::malloc_shared<int>(length, q);

        // inicializamos
        for (int i = 0; i < length; i++) {
            A[i] = i; //0,1,...,1023,1024
            B[i] = length - i; //1024,1023,...,1,0
        }

        // inicia cronómetro
        start = clock();

        // kernel
        q.parallel_for(sycl::range<1>{length}, [=](sycl::id<1> i) {
            C[i] = A[i] + B[i];
            });
        // esperamos que la GPU termine
        q.wait();

        // finaliza cronómetro
        end = clock();


        // presenta el resultado (solo imprime arreglo si hay max 1024 valores)
        if (length <= 1024) {
            for (int i = 0; i < length; i++) {
                printf("Resultados %d: (%d + %d = %d)\n", i, A[i], B[i], C[i]);
            }
        }
        // tiempo total de ejecución de la kernel
        double time_taken = double(end - start);
        printf("Clock ticks: %f\n", time_taken);

        // liberamos USM
        sycl::free(A, q);
        sycl::free(B, q);
        sycl::free(C, q);

    }
    catch (sycl::exception& e) {
        printf("Problemas !!!: %s\n", e.what());
        return 1;
    }

    return 0;
}

