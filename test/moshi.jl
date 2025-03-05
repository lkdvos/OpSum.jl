using Moshi.Data: @data

@data MyData{T} begin
    struct x
        a::T
    end
end

dat = MyData.x(1)
dat.a

@code_warntype dat.a
