#include "StringBuilder.h"

void StringBuilder::append(const std::string& s)
{
    _data.push_back(s);
    _length += s.size();
}

void StringBuilder::append(const char* s)
{
    std::string str(s);
    _length += str.size();
    _data.push_back(std::move(str));
}

void StringBuilder::append(char c)
{
    _data.emplace_back(1, c);
    _length += 1;
}

std::string StringBuilder::toString() const
{
    std::string result;
    result.reserve(_length);

    for (const auto& s : _data)
        result += s;

    return result;
}

void StringBuilder::clear()
{
    _data.clear();
    _length = 0;
}

size_t StringBuilder::length() const
{
    return _length;
}
