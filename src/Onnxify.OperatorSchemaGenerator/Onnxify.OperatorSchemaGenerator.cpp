#include "Onnxify.OperatorSchemaGenerator.h"
#include <onnx/defs/schema.h>

using namespace std;

int main()
{
    auto schemas = ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas();

    std::cout << "Total operators: " << schemas.size() << "\n\n";

    for (const auto& s : schemas)
    {
        std::cout << s.Name()
            << " (since opset "
            << s.SinceVersion()
            << ")\n";
    }

    return 0;
}

