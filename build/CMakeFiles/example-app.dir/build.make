# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

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
CMAKE_COMMAND = /opt/cmake-3.13.2-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.13.2-Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cbc/project/cpp-pytorch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cbc/project/cpp-pytorch/build

# Include any dependencies generated for this target.
include CMakeFiles/example-app.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/example-app.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/example-app.dir/flags.make

CMakeFiles/example-app.dir/example-app.cpp.o: CMakeFiles/example-app.dir/flags.make
CMakeFiles/example-app.dir/example-app.cpp.o: ../example-app.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cbc/project/cpp-pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/example-app.dir/example-app.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example-app.dir/example-app.cpp.o -c /home/cbc/project/cpp-pytorch/example-app.cpp

CMakeFiles/example-app.dir/example-app.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example-app.dir/example-app.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cbc/project/cpp-pytorch/example-app.cpp > CMakeFiles/example-app.dir/example-app.cpp.i

CMakeFiles/example-app.dir/example-app.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example-app.dir/example-app.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cbc/project/cpp-pytorch/example-app.cpp -o CMakeFiles/example-app.dir/example-app.cpp.s

# Object files for target example-app
example__app_OBJECTS = \
"CMakeFiles/example-app.dir/example-app.cpp.o"

# External object files for target example-app
example__app_EXTERNAL_OBJECTS =

example-app: CMakeFiles/example-app.dir/example-app.cpp.o
example-app: CMakeFiles/example-app.dir/build.make
example-app: ../libtorch/lib/libtorch.so
example-app: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
example-app: ../libtorch/lib/libc10.so
example-app: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
example-app: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
example-app: CMakeFiles/example-app.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cbc/project/cpp-pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example-app"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example-app.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/example-app.dir/build: example-app

.PHONY : CMakeFiles/example-app.dir/build

CMakeFiles/example-app.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/example-app.dir/cmake_clean.cmake
.PHONY : CMakeFiles/example-app.dir/clean

CMakeFiles/example-app.dir/depend:
	cd /home/cbc/project/cpp-pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cbc/project/cpp-pytorch /home/cbc/project/cpp-pytorch /home/cbc/project/cpp-pytorch/build /home/cbc/project/cpp-pytorch/build /home/cbc/project/cpp-pytorch/build/CMakeFiles/example-app.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/example-app.dir/depend

