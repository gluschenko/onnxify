#include <onnx/defs/schema.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <filesystem>

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
        op["domain"] = schema.domain();
        op["doc"] = schema.doc();

        // inputs
        op["inputs"] = json::array();

        for (const auto& input : schema.inputs())
        {
            json x;

            x["name"] = input.GetName();
            x["type"] = input.GetTypeStr();
            x["option"] = (int)input.GetOption();
            x["description"] = input.GetDescription();

            std::vector<std::string> types;
            types.reserve(input.GetTypes().size());

            for (const auto& type : input.GetTypes())
            {
                types.push_back(*type);
            }

            std::sort(types.begin(), types.end());

            for (const auto& type : types)
            {
                x["types"].push_back(type);
            }

            op["inputs"].push_back(x);
        }

        // outputs
        op["outputs"] = json::array();

        for (const auto& output : schema.outputs())
        {
            json x;

            x["name"] = output.GetName();
            x["type"] = output.GetTypeStr();
            x["option"] = (int)output.GetOption();
            x["description"] = output.GetDescription();

            std::vector<std::string> types;
            types.reserve(output.GetTypes().size());

            for (const auto& type : output.GetTypes())
            {
                types.push_back(*type);
            }

            std::sort(types.begin(), types.end());

            for (const auto& type : types)
            {
                x["types"].push_back(type);
            }

            op["outputs"].push_back(x);
        }

        // attributes
        op["attributes"] = json::array();

        for (const auto& attr : schema.attributes())
        {
            json x;

            x["name"] = attr.first;
            x["type"] = (int)attr.second.type;
            x["required"] = attr.second.required;

            if (!attr.second.description.empty())
            {
                x["description"] = attr.second.description;
            }

            op["attributes"].push_back(x);
        }

        root["operators"].push_back(op);
    }

    std::cout << "Total operators: " << schemas.size() << "\n\n";
    std::cout << root.dump(4) << std::endl;

    std::filesystem::path path = "../../../../Onnxify/Assets/onnx_operators.json";
    std::filesystem::create_directories(path.parent_path());

    std::ofstream file(path);
    file << root.dump(4);
    file.close();
    
    return 0;
}

