# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vladimir/Work/image_classification

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vladimir/Work/image_classification/build

# Include any dependencies generated for this target.
include CMakeFiles/image_classification.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/image_classification.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/image_classification.dir/flags.make

CMakeFiles/image_classification.dir/image_classification.cpp.o: CMakeFiles/image_classification.dir/flags.make
CMakeFiles/image_classification.dir/image_classification.cpp.o: ../image_classification.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vladimir/Work/image_classification/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/image_classification.dir/image_classification.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/image_classification.dir/image_classification.cpp.o -c /home/vladimir/Work/image_classification/image_classification.cpp

CMakeFiles/image_classification.dir/image_classification.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/image_classification.dir/image_classification.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vladimir/Work/image_classification/image_classification.cpp > CMakeFiles/image_classification.dir/image_classification.cpp.i

CMakeFiles/image_classification.dir/image_classification.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/image_classification.dir/image_classification.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vladimir/Work/image_classification/image_classification.cpp -o CMakeFiles/image_classification.dir/image_classification.cpp.s

# Object files for target image_classification
image_classification_OBJECTS = \
"CMakeFiles/image_classification.dir/image_classification.cpp.o"

# External object files for target image_classification
image_classification_EXTERNAL_OBJECTS =

image_classification: CMakeFiles/image_classification.dir/image_classification.cpp.o
image_classification: CMakeFiles/image_classification.dir/build.make
image_classification: /usr/local/lib/libopencv_gapi.so.4.6.0
image_classification: /usr/local/lib/libopencv_highgui.so.4.6.0
image_classification: /usr/local/lib/libopencv_ml.so.4.6.0
image_classification: /usr/local/lib/libopencv_objdetect.so.4.6.0
image_classification: /usr/local/lib/libopencv_photo.so.4.6.0
image_classification: /usr/local/lib/libopencv_stitching.so.4.6.0
image_classification: /usr/local/lib/libopencv_video.so.4.6.0
image_classification: /usr/local/lib/libopencv_videoio.so.4.6.0
image_classification: /home/vladimir/Work/libtorch/lib/libtorch.so
image_classification: /home/vladimir/Work/libtorch/lib/libc10.so
image_classification: /home/vladimir/Work/libtorch/lib/libkineto.a
image_classification: /usr/local/lib/libopencv_imgcodecs.so.4.6.0
image_classification: /usr/local/lib/libopencv_dnn.so.4.6.0
image_classification: /usr/local/lib/libopencv_calib3d.so.4.6.0
image_classification: /usr/local/lib/libopencv_features2d.so.4.6.0
image_classification: /usr/local/lib/libopencv_flann.so.4.6.0
image_classification: /usr/local/lib/libopencv_imgproc.so.4.6.0
image_classification: /usr/local/lib/libopencv_core.so.4.6.0
image_classification: /home/vladimir/Work/libtorch/lib/libc10.so
image_classification: CMakeFiles/image_classification.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/vladimir/Work/image_classification/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable image_classification"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/image_classification.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/image_classification.dir/build: image_classification

.PHONY : CMakeFiles/image_classification.dir/build

CMakeFiles/image_classification.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/image_classification.dir/cmake_clean.cmake
.PHONY : CMakeFiles/image_classification.dir/clean

CMakeFiles/image_classification.dir/depend:
	cd /home/vladimir/Work/image_classification/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vladimir/Work/image_classification /home/vladimir/Work/image_classification /home/vladimir/Work/image_classification/build /home/vladimir/Work/image_classification/build /home/vladimir/Work/image_classification/build/CMakeFiles/image_classification.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/image_classification.dir/depend

