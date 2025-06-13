CXX      := icpx
NVCC     := nvcc

EXTRACXXFLAGS := -Ofast -march=native -funroll-loops -flto -mavx2
CXXFLAGS      := -Wall -Wextra -Werror -fopenmp $(EXTRACXXFLAGS) -MMD -MP
#CUDAARCH      := -arch=sm_75
CUDAFLAGS     := -O3 --expt-relaxed-constexpr #-Xcompiler="$(EXTRACXXFLAGS)" #$(CUDAARCH) no es compatible con -flto
INCLUDE       := -I./src
CUINC         := -I/usr/local/cuda/include

CXXCMD := $(CXX) $(CXXFLAGS) $(INCLUDE)

# Fuentes y objetos
headers     := $(wildcard src/*.hpp)
sources     := $(wildcard src/*.cpp)
objects     := $(sources:src/%.cpp=src/%.o)
CU_SOURCES  := $(wildcard src/*.cu)
obj_cuda    := $(CU_SOURCES:src/%.cu=src/%.cuda.o)
deps        := $(objects:%.o=%.d) $(obj_cuda:%.cuda.o=%.d)

mains = graphics/burned_probabilities_data graphics/fire_animation_data

all: $(mains)

src/%.o: src/%.cpp $(headers)
	$(CXXCMD) -c $< -o $@


src/%.cuda.o: src/%.cu $(headers)
	$(NVCC) $(CUINC) $(CUDAFLAGS) -c $< -o $@

# usamos icpx y enlazamos cudart
LDFLAGS := -L/usr/local/cuda/lib64 -lcudart -qopenmp -lcudadevrt
$(mains): %: %.cpp $(objects) $(obj_cuda)
	$(CXXCMD) $^ $(LDFLAGS) -o $@

data.zip:
	wget https://cs.famaf.unc.edu.ar/~nicolasw/data.zip

data: data.zip
	unzip data.zip

clean:
	rm -f $(objects) $(obj_cuda) $(mains) data.zip
	rm -f $(deps)

-include $(deps)

.PHONY: all clean data