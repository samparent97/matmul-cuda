add_test( [==[MMTest/MMTest.TestSingleRowDecomp/(512,512,512,64,32,8,1,"SameSizePwrOf2")]==] /root/sam/matmul-cuda/build/test/lab4_mm_test [==[--gtest_filter=MMTest/MMTest.TestSingleRowDecomp/0]==] --gtest_also_run_disabled_tests)
set_tests_properties( [==[MMTest/MMTest.TestSingleRowDecomp/(512,512,512,64,32,8,1,"SameSizePwrOf2")]==] PROPERTIES WORKING_DIRECTORY /root/sam/matmul-cuda/build/test SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
add_test( [==[MMTest/MMTest.Test1DTile/(512,512,512,64,32,8,1,"SameSizePwrOf2")]==] /root/sam/matmul-cuda/build/test/lab4_mm_test [==[--gtest_filter=MMTest/MMTest.Test1DTile/0]==] --gtest_also_run_disabled_tests)
set_tests_properties( [==[MMTest/MMTest.Test1DTile/(512,512,512,64,32,8,1,"SameSizePwrOf2")]==] PROPERTIES WORKING_DIRECTORY /root/sam/matmul-cuda/build/test SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
add_test( [==[MMTest/MMTest.Test2DTile/(512,512,512,64,32,8,1,"SameSizePwrOf2")]==] /root/sam/matmul-cuda/build/test/lab4_mm_test [==[--gtest_filter=MMTest/MMTest.Test2DTile/0]==] --gtest_also_run_disabled_tests)
set_tests_properties( [==[MMTest/MMTest.Test2DTile/(512,512,512,64,32,8,1,"SameSizePwrOf2")]==] PROPERTIES WORKING_DIRECTORY /root/sam/matmul-cuda/build/test SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set( lab4_mm_test_TESTS [==[MMTest/MMTest.TestSingleRowDecomp/(512,512,512,64,32,8,1,"SameSizePwrOf2")]==] [==[MMTest/MMTest.Test1DTile/(512,512,512,64,32,8,1,"SameSizePwrOf2")]==] [==[MMTest/MMTest.Test2DTile/(512,512,512,64,32,8,1,"SameSizePwrOf2")]==])