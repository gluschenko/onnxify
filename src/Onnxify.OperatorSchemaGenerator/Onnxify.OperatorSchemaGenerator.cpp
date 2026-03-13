#include "Onnxify.OperatorSchemaGenerator.h"
#include <onnx/defs/schema.h>
#include "StringBuilder.h"
#include <string>
#include <vector>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

struct Operator
{
public:
    string name;
    int sinceVersion = 0;
};

int main()
{
    auto builder = new StringBuilder();

    auto schemas = ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas();
    auto operators = std::vector<Operator>();

    std::cout << "Total operators: " << schemas.size() << "\n\n";

    for (const auto& item : schemas)
    {
        auto& attributes = item.attributes();
        auto& inputs = item.inputs();
        auto& outputs = item.outputs();

		json j;


        Operator op;
        op.name = item.Name();
        op.sinceVersion = item.SinceVersion();

        operators.push_back(op);
    }

    builder->append("{ ");
    builder->append("operators: ");
    builder->append("[");

    for (const auto& item : operators)
    {
        builder->append(R"(
        
        )");
    }

    builder->append("]");
    builder->append(" }");

    auto text = builder->toString();

    std::cout << "Source code: " << text << "\n\n";

    return 0;
}

