#pragma once

#include <vector>
#include <string>

class StringBuilder
{
private:
    std::vector<std::string> _data;
    size_t _length = 0;

public:
    void append(const std::string& s);
    void append(const char* s);
    void append(char c);

    std::string toString() const;

    void clear();
    size_t length() const;
};
