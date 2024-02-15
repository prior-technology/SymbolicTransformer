using Test
using PythonCall

@testset "PythonCallTest" begin
    re = pyimport("re")
    words = re.findall("[a-zA-Z]+", "Be careful") 
    @test pyconvert(String, words[0]) = "Be"
    
end
