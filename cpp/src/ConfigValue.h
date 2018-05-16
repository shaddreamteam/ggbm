#ifndef CPP_CONFIGVALUE_H
#define CPP_CONFIGVALUE_H

#include <unordered_map>
#include "Base.h"


enum ParamType {
    kUint32_t,
    kFloatType,
    kString,
    kBool,
    kObjectiveType,
};


class ConfigValue {
public:
    ConfigValue(const std::string& s, ParamType type);

    operator uint32_t() const;

    operator float_type() const;

    operator std::string() const;

    operator ObjectiveType() const;

    operator bool() const;

    ParamType GetType();

private:
    ObjectiveType StringToObj(const std::string& s);

    bool StringToBool(const std::string& s);

    uint32_t val_int_;
    float_type val_float_;
    std::string val_string_;
    ObjectiveType val_objective_;
    bool val_bool_;
    ParamType type_; 
};



#endif //CPP_CONFIGVALUE_H
