using FFTW

"""
    dst1u(x, dims=1)

performs the unitary (i.e., L^2 norm-preserving) version of the DST Type I.
Note that this is a self-inverse function, so that `x == dst1u(dst1u(x))`.

### Input Arguments
* `x`: An input real-valued array
* `dims`: direction in `x` for DST-I to apply; the default value is `1`, i.e., the first index direction of the input array.

### Output Argument
* An array of the same size and shape of the input array `x` whose `dims` index direction contain their DST-I coefficients.
"""

function dst1u(x, dims=1)
    N = size(x, dims)
    return FFTW.r2r(x, FFTW.RODFT00, dims)/sqrt(2*(N+1))
end


"""
    dst1u!(x, dims=1)

performs the in-place and unitary (i.e., L^2 norm-preserving) version of the DST Type I. Note that this is a self-inverse function, so that `x === dst1u!(dst1u!(x))`.

### Input Arguments
    * `x`: An input real-valued array
    * `dims`: direction in `x` for DST-I to apply; the default value is `1`, i.e., the first index direction of the input array.
        
"""

function dst1u!(x, dims=1)
    N = size(x, dims)
    FFTW.r2r!(x, FFTW.RODFT00, dims)
    x ./= sqrt(2*(N+1)) # This "." is critical in this in-place case!!
end
