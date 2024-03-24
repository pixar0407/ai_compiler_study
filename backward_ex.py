import torch
import triton
import triton.language as tl
from from_TransformerEngine import *
from torch.autograd import   Variable

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def kerenl_rotate_half_back(x_ptr,  # *Pointer* to first input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               d_half: tl.constexpr,
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0,d_half)

    output_ptr_half = output_ptr + d_half
    x_ptr_half      = x_ptr      + d_half

    mask = offsets < n_elements

    half_1 = tl.load(x_ptr + offsets, mask=mask)
    half_2 = tl.load(x_ptr_half + offsets, mask=mask)

    half_1 = (-half_1)

    tl.store(output_ptr + offsets, half_2, mask=mask)
    tl.store(output_ptr_half + offsets, half_1, mask=mask)

@triton.jit
def arith_kernel(x_ptr,  # *Pointer* to first input vector.
               half_ptr,  # *Pointer* to second input vector.
               freqs_ptr,
               output1_ptr,  # *Pointer* to output vector.
               output2_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               LARGE_BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    
    pid = tl.program_id(axis=0)
    large_block_start = pid * LARGE_BLOCK_SIZE
    large_offsets = large_block_start + tl.arange(0, LARGE_BLOCK_SIZE)
    large_mask = large_offsets < n_elements

    x = tl.load(x_ptr + large_offsets, mask=large_mask)
    half = tl.load(half_ptr + large_offsets, mask=large_mask)
    freqs = tl.load(freqs_ptr + large_offsets, mask=large_mask)

    cos_ = tl.cos(freqs).to(x.dtype)
    sin_ = tl.sin(freqs).to(x.dtype)

    tl.store(output1_ptr + large_offsets, (half * sin_), mask=large_mask)
    tl.store(output2_ptr + large_offsets, (x * cos_), mask=large_mask)

def rope_bw(x: torch.Tensor, freqs: torch.Tensor):
    output_rotate_half = torch.empty_like(x)
    output1 = torch.empty_like(x)
    output2 = torch.empty_like(x)
    output = torch.empty_like(x)

    assert x.is_cuda and freqs.is_cuda and output_rotate_half.is_cuda and output.is_cuda

    s,b,h,d = x.shape
    n_elements = s*d
    n_elements_arith = s*b*h*d

    grid = lambda meta: (triton.cdiv(n_elements_arith, meta['BLOCK_SIZE']), )
    kerenl_rotate_half_back[grid](x, output_rotate_half, n_elements_arith,d_half=int(d/2), BLOCK_SIZE=d)
         
    freqs = freqs.repeat(1,b,h,1) #주석 3 
    grid_arith = lambda meta: (triton.cdiv(n_elements_arith, meta['LARGE_BLOCK_SIZE']), )
    grid_add = lambda meta: (triton.cdiv(n_elements_arith, meta['BLOCK_SIZE']), )
    arith_kernel[grid_arith](x, output_rotate_half, freqs, output1, output2, n_elements_arith, LARGE_BLOCK_SIZE=b*h*d) # 주석 2 
    add_kernel[grid_add](output1, output2, output, n_elements_arith, BLOCK_SIZE=b*h*d) # 주석 1
    return output


####################################
#################################### inputs for BW
####################################
torch.manual_seed(5468)
s=16
b=8
h=4
d=2
d_half = int(d/2)

intput_to_TE = Variable(torch.rand((s,b,h,d), device='cuda:0'),requires_grad=True)
loss_trion = Variable(torch.rand((s,b,h,d), device='cuda:0'),requires_grad=True)
freqs_half  = torch.rand(s,1,1,d_half) 
freqs = torch.concat((freqs_half,freqs_half),dim=-1).to(device='cuda:0')

x_grad_triton = rope_bw(loss_trion,freqs) #.to(torch.half)

loss_TE = loss_trion.clone().detach().to(device='cuda:0')
loss_TE = Variable(loss_TE,requires_grad=True)

output_te = torch.tensor([],requires_grad=True)
output_te.retain_grad()
output_te = apply_rotary_pos_emb(intput_to_TE,freqs) #.to(torch.half)
output_te.retain_grad()
 
grad_te=output_te.backward(loss_TE)
x_grad_te = intput_to_TE.grad

####################################
#################################### inputs for BW
####################################
"""
error = torch.abs(x_grad_te - x_grad_triton)
error_max = torch.max(error)
if error_max == 0.0:
    print("No Error")
else:
    print("Error occurs! Max Error :",error_max)

torch.testing.assert_close(x_grad_triton, x_grad_te)
"""

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(3, 9, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):

    ####################################
    #################################### inputs for BW
    ####################################
    
    b=8
    h=16
    d=512
    
    intput_to_TE = Variable(torch.rand((size,b,h,d), device='cuda:0'),requires_grad=True)
    loss_trion = Variable(torch.rand((size,b,h,d), device='cuda:0'),requires_grad=True)
    freqs_half  = torch.rand(size,1,1,int(d/2)) 
    freqs = torch.concat((freqs_half,freqs_half),dim=-1).to(device='cuda:0')
    
    x_grad_triton = rope_bw(loss_trion,freqs) #.to(torch.half)
    
    loss_TE = loss_trion.clone().detach().to(device='cuda:0')
    loss_TE = Variable(loss_TE,requires_grad=True)
    
    output_te = torch.tensor([],requires_grad=True)
    output_te.retain_grad()
    output_te = apply_rotary_pos_emb(intput_to_TE,freqs) #.to(torch.half)
    output_te.retain_grad()
     
    ####################################
    #################################### inputs for BW
    ####################################

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_bw(loss_trion,freqs), quantiles=quantiles)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: output_te.backward(loss_TE,retain_graph=True), quantiles=quantiles)
    gbps = lambda ms: b*h*d * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=False)
