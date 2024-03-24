import torch
import triton
import triton.language as tl
from from_TransformerEngine import *
from torch.autograd import Variable


@triton.jit
def kerenl_rotate_half(x_ptr,  # *Pointer* to first input vector.
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

    half_2 = tl.load(x_ptr_half + offsets, mask=mask)
    tl.store(output_ptr + offsets, (-half_2), mask=mask)

    half_1 = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr_half + offsets, half_1, mask=mask)


@triton.jit
def arith_kernel(x_ptr,  # *Pointer* to first input vector.
               half_ptr,  # *Pointer* to second input vector.
               freqs_ptr,
               output_ptr,  # *Pointer* to output vector.
               freq_repeat_size,
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    
    pid = tl.program_id(axis=0)

    freq_pid = pid // freq_repeat_size
    freq_block_start = freq_pid * BLOCK_SIZE
    freq_offsets = freq_block_start + tl.arange(0, BLOCK_SIZE)
    freq_mask = freq_offsets < n_elements

    freqs = tl.load(freqs_ptr + freq_offsets, mask=freq_mask)
    cos_ = tl.cos(freqs)
    sin_ = tl.sin(freqs)

    large_block_start = pid * BLOCK_SIZE
    large_offsets = large_block_start + tl.arange(0, BLOCK_SIZE)
    large_mask = large_offsets < n_elements

    x = tl.load(x_ptr + large_offsets, mask=large_mask)
    half = tl.load(half_ptr + large_offsets, mask=large_mask)
    tl.store(output_ptr + large_offsets, (x * cos_)+(half * sin_), mask=large_mask)

def rope_fw(x: torch.Tensor, freqs: torch.Tensor):
    output_rotate_half = torch.empty_like(x)
    output = torch.empty_like(x)

    assert x.is_cuda and freqs.is_cuda and output_rotate_half.is_cuda and output.is_cuda 

    s,b,h,d = x.shape
    n_elements_arith = s*b*h*d

    grid = lambda meta: (triton.cdiv(n_elements_arith, meta['BLOCK_SIZE']), )
    kerenl_rotate_half[grid](x, output_rotate_half, n_elements_arith,d_half=int(d/2), BLOCK_SIZE=d)
         
    #freqs = freqs.repeat(1,b,h,1)
    freq_repeat_size = b*h
    arith_kernel[grid](x, output_rotate_half, freqs,output, freq_repeat_size,n_elements_arith, BLOCK_SIZE=d)
    return output



torch.manual_seed(5468)
s=16
b=8
h=4
d=2
d_half = int(d/2)

x = Variable(torch.rand((s,b,h,d), device='cuda:0'),requires_grad=True)
freqs_half  = torch.rand(s,1,1,d_half) 
freqs = torch.concat((freqs_half,freqs_half),dim=-1).to(device='cuda:0')


output_triton = rope_fw(x,freqs) #.to(torch.half)
output_te = apply_rotary_pos_emb(x,freqs) #.to(torch.half)

"""
error = torch.abs(output_te - output_triton)
error_max = torch.max(error)
if error_max == 0.0:
    print("No Error")
else:
    print("Error occurs! Max Error :",error_max)
"""
    
#assert torch.allclose(output_triton, output_te), (output_triton, output_te)
torch.testing.assert_close(output_triton, output_te)


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
        plot_name='performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):

    b=8
    h=16
    d=512
    x = Variable(torch.rand((size,b,h,d), device='cuda:0'),requires_grad=True)
    freqs_half  = torch.rand(size,1,1,int(d/2)) 
    freqs = torch.concat((freqs_half,freqs_half),dim=-1).to(device='cuda:0')

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: apply_rotary_pos_emb(x, freqs), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_fw(x,freqs), quantiles=quantiles)
    gbps = lambda ms: size*b*h*d / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=False)
