# intel-sgx-text-summarization-overhead
### Important
1. The only way to reproduce our work is using 
```
Microsoft Azure Standard DC8 v2 (8 vcpus, 32 GiB memory), Ubuntu Server 20.04)
```
Please consider that if you use different config, like lower memory, you need to change other configs as well like enclave size.

3. It is impossible to use our work in a Jupyter Notebook since Gramine LibOS can run only one single .py file. Therefore, adding a Jupyter Notebook doesn't help you to understand or reproduce our work.

4. Attaching our VM does not help you to reproduce our work at all since all of configurations and setups are platform dependent, considering the fact that when you install them using apt or pip, then a specific version that is match with your hardware will be installed.

# Installation and Dependencies

## Intel SGX Installation
1. Make sure that you have access to an Intel CPU
2. Make sure the CPU supports SGX (Even some Intel CPUs do not support SGX). One way to do that is running `cpuid | grep -i SGX`
3. Make sure the CPU supports FLC (Flexible Launch Control). 
4. It's better to use the latest version of BIOS. So, if there is a newer version, update your BIOS. Note: updating BIOS may home effect on GRUB.
5. Activate SGX in BIOS.
6. Disable secure boot in BIOS.
7. Install Intel SGX Driver for Linux. Follow the instructions in this link: https://github.com/intel/linux-sgx
8. Install Intel SGX SDK and PSW. Link: https://github.com/intel/linux-sgx#build-the-intelr-sgx-sdk-and-intelr-sgx-psw-package
9. Make sure that aesmd.service is up and running: `systemctl status aesmd.service`
10. Make sure that everything is up and running by executing the sample. If it successfully compiles, executes, and shows proper output, then you can continue next steps. Otherwise, next steps does not work. Link: https://github.com/intel/linux-sgx#test-the-intelr-sgx-sdk-package-with-the-code-samples
13. Make sure that `/dev/sgx_enclave` is created.

## Gramine Installation
To be able to execute python codes without need to modify it for SGX compatibility, there are two solutions. The first solution is using SCONE (https://sconedocs.github.io/) and the second solution is using Gramine, which is the also known as Graphine (https://gramine.readthedocs.io/). SCONE uses contanarization techniques and docker to convert a code to a docker image. Then, they run the created docker image on top of their own image which is compatible with SGX. We may use SCONE for the experiments part, but since gramine was simpler, we chose it first.
1. Follow https://gramine.readthedocs.io/en/latest/quickstart.html
2. You have to be able successfully execute `CI-Examples/helloworld` with and without SGX. If there are any errors, fix them before moving forward.
3. To get an idea how Gramine helps use to train and evaluate a model, this article is suggested: https://gramine.readthedocs.io/en/latest/tutorials/pytorch/index.html

## Python, Hugging face, Transformers, other
Hugging face is a platform that is publicly available for everyone. Companies or individuals publish their latest models (algorithms) in that platform. Other people can simply go through their platform, select a trained model, and evaluate anything that they want. To be able to run these models, you need to have `Python 3.7+`. Additionally, these libraries are required: `tensorflow jax jaxlib transformers flax`. You can install them by either `conda` or `pip`.
This project is based on `conda`. `environment.yml` file can be used to generate an environment automatically with all required dependencies. Please note that some of these libraries are not required, however, by using `environment.yml` you can be sure that all libraries will be available.
Therefore, we suggest using this command: 
```
conda env create -f environment.yml
```

## How to Run
Gramine Direct mode:
```
time gramine-direct <script-name>
```
Gramine SGX mode:
```
time graminde-sgx <script-name>
```
