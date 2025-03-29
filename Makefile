CXX = clang++
EXTRACXXFLAGS =
DEFINES =
CXXFLAGS = -Wall -Wextra -Werror -std=c++17 -fopenmp
LDFLAGS = -L/opt/AMD/aocc-compiler-4.0.0/lib -lomp
INCLUDE = -I./src -I/usr/lib/gcc/x86_64-linux-gnu/11/include
CXXCMD = $(CXX) $(DEFINES) $(CXXFLAGS) $(INCLUDE)

headers = $(wildcard ./src/*.hpp)
sources = $(wildcard ./src/*.cpp)
objects_names = $(sources:./src/%.cpp=%)
objects = $(objects_names:%=./src/%.o)

mains = graphics/burned_probabilities_data graphics/fire_animation_data

all: $(mains)

%.o: %.cpp $(headers)
	$(CXXCMD) -c $< -o $@

$(mains): %: %.cpp $(objects) $(headers)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< $(objects) $(LDFLAGS) -o $@

data.zip:
	wget https://cs.famaf.unc.edu.ar/~nicolasw/data.zip

data: data.zip
	unzip data.zip

clean:
	rm -f $(objects) $(mains)

.PHONY: all clean
