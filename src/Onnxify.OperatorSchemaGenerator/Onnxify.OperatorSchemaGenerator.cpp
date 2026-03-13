#include "Onnxify.OperatorSchemaGenerator.h"
#include <onnx/defs/schema.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

using namespace std;
using json = nlohmann::json;

int main()
{
    json root;
    root["operators"] = json::array();

    auto schemas = ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas();

    for (const auto& schema : schemas)
    {
        json op;

        op["name"] = schema.Name();
        op["sinceVersion"] = schema.SinceVersion();

        // inputs
        op["inputs"] = json::array();

        for (const auto& input : schema.inputs())
        {
            json j;

            j["name"] = input.GetName();
            j["type"] = input.GetTypeStr();
            j["option"] = (int)input.GetOption();

            op["inputs"].push_back(j);
        }

        // outputs
        op["outputs"] = json::array();

        for (const auto& output : schema.outputs())
        {
            json j;

            j["name"] = output.GetName();
            j["type"] = output.GetTypeStr();
            j["option"] = (int)output.GetOption();

            op["outputs"].push_back(j);
        }

        // attributes
        op["attributes"] = json::array();

        for (const auto& attr : schema.attributes())
        {
            json j;

            j["name"] = attr.first;
            j["type"] = (int)attr.second.type;
            j["required"] = attr.second.required;

            if (!attr.second.description.empty())
                j["description"] = attr.second.description;

            op["attributes"].push_back(j);
        }

        root["operators"].push_back(op);
    }

    std::cout << "Total operators: " << schemas.size() << "\n\n";
    std::cout << root.dump(4) << std::endl;

    std::ofstream file("onnx_operators.json");
    file << root.dump(4);
    file.close();
    
    return 0;
}

