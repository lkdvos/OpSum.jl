using Literate: Literate
using OpSum: OpSum

Literate.markdown(
  joinpath(pkgdir(OpSum), "examples", "README.jl"),
  joinpath(pkgdir(OpSum), "docs", "src");
  flavor=Literate.DocumenterFlavor(),
  name="index",
)
