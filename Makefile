# Makefile for project3

# Compiler settings
CXX = g++

# Compiler flags
CXXFLAGS = $(shell pkg-config --cflags opencv4)
# Linker flags
LDFLAGS = $(shell pkg-config --libs opencv4)

# Source files
SOURCES = main.cpp helpers.cpp 
# Object files
OBJECTS = $(SOURCES:.cpp=.o)
# Executable name
EXEC = prj3

# Default target
all: $(EXEC)

$(EXEC): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(EXEC) $(LDFLAGS)

# To obtain object files
%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

# To remove generated files
clean:
	rm -f $(EXEC) $(OBJECTS)