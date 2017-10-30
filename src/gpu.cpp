// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <assert.h>
#include <vector>
#include "gpu.h"

//#ifdef _OPENCL
#include <CL/cl.h>
//#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define OPENCL_CHECK_ERRORS(ERR)                                                  \
if(ERR != CL_SUCCESS)                                                             \
{                                                                                 \
    printf                                                                          \
    (                                                                             \
        "OpenCL error with code %s happened in file %s at line %d. Exiting.\n",   \
        opencl_error_to_str(ERR), __FILE__, __LINE__                              \
    );                                                                            \
    assert(false);                                                                \
    return -1;                                                                       \
}

/* This function helps to create informative messages in
* case when OpenCL errors occur. The function returns a string
* representation for an OpenCL error code.
* For example, "CL_DEVICE_NOT_FOUND" instead of "-1".
*/
const char* opencl_error_to_str(cl_int error)
{
#define CASE_CL_CONSTANT(NAME) case NAME: return #NAME;

    // Suppose that no combinations are possible.
    switch (error)
    {
        CASE_CL_CONSTANT(CL_SUCCESS)
        CASE_CL_CONSTANT(CL_DEVICE_NOT_FOUND)
        CASE_CL_CONSTANT(CL_DEVICE_NOT_AVAILABLE)
        CASE_CL_CONSTANT(CL_COMPILER_NOT_AVAILABLE)
        CASE_CL_CONSTANT(CL_MEM_OBJECT_ALLOCATION_FAILURE)
        CASE_CL_CONSTANT(CL_OUT_OF_RESOURCES)
        CASE_CL_CONSTANT(CL_OUT_OF_HOST_MEMORY)
        CASE_CL_CONSTANT(CL_PROFILING_INFO_NOT_AVAILABLE)
        CASE_CL_CONSTANT(CL_MEM_COPY_OVERLAP)
        CASE_CL_CONSTANT(CL_IMAGE_FORMAT_MISMATCH)
        CASE_CL_CONSTANT(CL_IMAGE_FORMAT_NOT_SUPPORTED)
        CASE_CL_CONSTANT(CL_BUILD_PROGRAM_FAILURE)
        CASE_CL_CONSTANT(CL_MAP_FAILURE)
        CASE_CL_CONSTANT(CL_MISALIGNED_SUB_BUFFER_OFFSET)
        CASE_CL_CONSTANT(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
        CASE_CL_CONSTANT(CL_INVALID_VALUE)
        CASE_CL_CONSTANT(CL_INVALID_DEVICE_TYPE)
        CASE_CL_CONSTANT(CL_INVALID_PLATFORM)
        CASE_CL_CONSTANT(CL_INVALID_DEVICE)
        CASE_CL_CONSTANT(CL_INVALID_CONTEXT)
        CASE_CL_CONSTANT(CL_INVALID_QUEUE_PROPERTIES)
        CASE_CL_CONSTANT(CL_INVALID_COMMAND_QUEUE)
        CASE_CL_CONSTANT(CL_INVALID_HOST_PTR)
        CASE_CL_CONSTANT(CL_INVALID_MEM_OBJECT)
        CASE_CL_CONSTANT(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        CASE_CL_CONSTANT(CL_INVALID_IMAGE_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_SAMPLER)
        CASE_CL_CONSTANT(CL_INVALID_BINARY)
        CASE_CL_CONSTANT(CL_INVALID_BUILD_OPTIONS)
        CASE_CL_CONSTANT(CL_INVALID_PROGRAM)
        CASE_CL_CONSTANT(CL_INVALID_PROGRAM_EXECUTABLE)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL_NAME)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL_DEFINITION)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL)
        CASE_CL_CONSTANT(CL_INVALID_ARG_INDEX)
        CASE_CL_CONSTANT(CL_INVALID_ARG_VALUE)
        CASE_CL_CONSTANT(CL_INVALID_ARG_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL_ARGS)
        CASE_CL_CONSTANT(CL_INVALID_WORK_DIMENSION)
        CASE_CL_CONSTANT(CL_INVALID_WORK_GROUP_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_WORK_ITEM_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_GLOBAL_OFFSET)
        CASE_CL_CONSTANT(CL_INVALID_EVENT_WAIT_LIST)
        CASE_CL_CONSTANT(CL_INVALID_EVENT)
        CASE_CL_CONSTANT(CL_INVALID_OPERATION)
        CASE_CL_CONSTANT(CL_INVALID_GL_OBJECT)
        CASE_CL_CONSTANT(CL_INVALID_BUFFER_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_MIP_LEVEL)
        CASE_CL_CONSTANT(CL_INVALID_GLOBAL_WORK_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_PROPERTY)

        default:
            return "UNKNOWN ERROR CODE";
    }
#undef CASE_CL_CONSTANT
}

struct OpenCLObject
{
    // Regular OpenCL objects:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
};

OpenCLObject *g_pOpenCLObject = NULL;

const char *kernelstring =
"__kernel void myGEMM1(const int M, const int N, const int K,"
"                      const __global float* A,"
"                      const __global float* B,"
"                      __global float* C) {"
"    const int globalRow = get_global_id(0);"
"    const int globalCol = get_global_id(1);"
"    float acc = 0.0f;"
"    for (int k=0; k<K; k++) {"
"        acc += A[k*M + globalRow] * B[globalCol*K + k];"
"    }"
"    C[globalCol*M + globalRow] = acc;"
"}";

namespace ncnn {
    int init_OpenCL() {

        assert(g_pOpenCLObject == NULL);
        g_pOpenCLObject = new OpenCLObject();
        memset(g_pOpenCLObject, 0, sizeof(OpenCLObject));

        cl_int err = CL_SUCCESS;
        printf("start init_OpenCL...\n");
        
        // STEP 1: init platforms
        cl_uint num_of_platforms = 0;
        err = clGetPlatformIDs(0, 0, &num_of_platforms);
        OPENCL_CHECK_ERRORS(err);

        std::vector<cl_platform_id> platforms(num_of_platforms);
        err = clGetPlatformIDs(num_of_platforms, &platforms[0], 0);
        OPENCL_CHECK_ERRORS(err);

        // STEP 2: init devices
        cl_uint selected_platform_index = num_of_platforms;

        for (cl_uint i = 0; i < num_of_platforms; ++i)
        {
            cl_uint num_devices = 0;
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, 0, &num_devices);
            if (err != CL_DEVICE_NOT_FOUND) {
                // Handle all other type of errors from clGetDeviceIDs here
                OPENCL_CHECK_ERRORS(err);
                assert(num_devices > 0);
                g_pOpenCLObject->platform = platforms[i];
                break;
            }
        }

        
        // STEP 3: create context
        cl_context_properties context_props[] = {
            CL_CONTEXT_PLATFORM, cl_context_properties(g_pOpenCLObject->platform),
            0
        };

        g_pOpenCLObject->context = clCreateContextFromType (context_props, CL_DEVICE_TYPE_GPU, 0, 0, &err);
        OPENCL_CHECK_ERRORS(err);

        err = clGetContextInfo(g_pOpenCLObject->context, CL_CONTEXT_DEVICES, sizeof(g_pOpenCLObject->device),
                &g_pOpenCLObject->device, 0);
        OPENCL_CHECK_ERRORS(err);

        // STEP 4: create and compile program.
        g_pOpenCLObject->program = clCreateProgramWithSource(g_pOpenCLObject->context, 1, &kernelstring, NULL, NULL);
        err = clBuildProgram(g_pOpenCLObject->program, 0, NULL, "", NULL, NULL);
        OPENCL_CHECK_ERRORS(err);

        // STEP 5: create queue
        g_pOpenCLObject->queue = clCreateCommandQueue (g_pOpenCLObject->context, g_pOpenCLObject->device, 
            CL_QUEUE_PROFILING_ENABLE,&err
        );
        OPENCL_CHECK_ERRORS(err);
        printf("initOpenCL finished successfully\n");

        return 0;
    }

    int uninit_OpenCL() {

        // g_pOpenCLObject must be created before uninit.
        assert(g_pOpenCLObject != NULL);

        if (g_pOpenCLObject->queue != NULL) {
            clReleaseCommandQueue(g_pOpenCLObject->queue);
            g_pOpenCLObject->queue = NULL;
        }
        
        if (g_pOpenCLObject->context != NULL) {
            clReleaseContext(g_pOpenCLObject->context);
            g_pOpenCLObject->context = NULL;
        }

        if (g_pOpenCLObject->program != NULL) {
            clReleaseProgram(g_pOpenCLObject->program);
            g_pOpenCLObject->program = NULL;
        }
        
        if (g_pOpenCLObject != NULL)
        {
            delete g_pOpenCLObject;
            g_pOpenCLObject = NULL;
        }
        return 0;
    }

    OpenCLObject *get_OpenCL_object(){
        return g_pOpenCLObject;
    }

} // namespace ncnn
