#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <core/session/onnxruntime_c_api.h>
#include <utility>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <onnx/defs/schema.h>

using namespace std;
using json = nlohmann::json;
using namespace onnx;

int main()
{
    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv* env = nullptr;
    OrtStatus* st = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "schema_dump", &env);
    if (st != nullptr) {
        const char* msg = ort->GetErrorMessage(st);
        std::cerr << "CreateEnv failed: " << (msg ? msg : "") << "\n";
        ort->ReleaseStatus(st);
        return 1;
    }

    json root;
    root["operators"] = json::array();

    auto schemas = ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas_with_history();

    std::sort(
        schemas.begin(),
        schemas.end(),
        [](const ONNX_NAMESPACE::OpSchema& left, const ONNX_NAMESPACE::OpSchema& right)
        {
            if (left.domain() != right.domain())
            {
                return left.domain() < right.domain();
            }

            if (left.Name() != right.Name())
            {
                return left.Name() < right.Name();
            }

            return left.SinceVersion() < right.SinceVersion();
        }
    );

    for (const auto& schema : schemas)
    {
        json op;

        const char* doc = schema.doc();

        op["name"] = schema.Name();
        op["sinceVersion"] = schema.SinceVersion();
        op["domain"] = schema.domain();
        op["doc"] = doc != nullptr ? schema.doc() : "";

        // inputs
        op["inputs"] = json::array();

        for (const auto& input : schema.inputs())
        {
            json x;

            x["name"] = input.GetName();
            x["type"] = input.GetTypeStr();
            x["option"] = (int)input.GetOption();
            x["minArity"] = input.GetMinArity();
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
            x["minArity"] = output.GetMinArity();
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

        std::vector<std::string> attr_names;
        for (const auto& kv : schema.attributes())
        {
            attr_names.push_back(kv.first);
        }

        std::sort(attr_names.begin(), attr_names.end());

        for (const auto& name : attr_names)
        {
            const auto& value = schema.attributes().at(name);

            json x;

            x["name"] = name;
            x["type"] = (int)value.type;
            x["required"] = value.required;

            if (!value.description.empty())
            {
                x["description"] = value.description;
            }

            const ONNX_NAMESPACE::AttributeProto& def = value.default_value;
            const auto type = def.type();

            switch (type)
            {
            case ONNX_NAMESPACE::AttributeProto::INT:
                x["default"] = def.i();
                break;

            case ONNX_NAMESPACE::AttributeProto::FLOAT:
                x["default"] = def.f();
                break;

            case ONNX_NAMESPACE::AttributeProto::STRING:
                x["default"] = def.s();
                break;

            case ONNX_NAMESPACE::AttributeProto::INTS:
            {
                json arr = json::array();
                for (auto v : def.ints())
                {
                    arr.push_back(v);
                }
                x["default"] = arr;
                break;
            }

            case ONNX_NAMESPACE::AttributeProto::FLOATS:
            {
                json arr = json::array();
                for (auto v : def.floats())
                {
                    arr.push_back(v);
                }
                x["default"] = arr;
                break;
            }

            case ONNX_NAMESPACE::AttributeProto::STRINGS:
            {
                json arr = json::array();
                for (const auto& v : def.strings())
                {
                    arr.push_back(v);
                }
                x["default"] = arr;
                break;
            }

            case ONNX_NAMESPACE::AttributeProto::TENSOR:
                x["default"] = "<TENSOR>";
                break;

            case ONNX_NAMESPACE::AttributeProto::TENSORS:
                x["default"] = "<TENSORS>";
                break;

            case ONNX_NAMESPACE::AttributeProto::GRAPH:
                x["default"] = "<GRAPH>";
                break;

            case ONNX_NAMESPACE::AttributeProto::GRAPHS:
                x["default"] = "<GRAPHS>";
                break;

            case ONNX_NAMESPACE::AttributeProto::SPARSE_TENSOR:
                x["default"] = "<SPARSE_TENSOR>";
                break;

            case ONNX_NAMESPACE::AttributeProto::SPARSE_TENSORS:
                x["default"] = "<SPARSE_TENSORS>";
                break;

            default:
                break;
            }

            op["attributes"].push_back(x);
        }

        root["operators"].push_back(op);
    }

    std::cout << "Total operator schemas: " << schemas.size() << "\n\n";
    std::cout << root.dump(4) << std::endl;

    std::filesystem::path path = "../../../../Onnxify/Assets/onnx_operators.json";
    std::filesystem::create_directories(path.parent_path());

    std::ofstream file(path);
    file << root.dump(4);
    file.close();
    
    return 0;
}
