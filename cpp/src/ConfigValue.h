#ifndef CPP_CONFIGVALUE_H
#define CPP_CONFIGVALUE_H

#include <unordered_map>
#include "Base.h"


enum ParamType {
    kUint32_t,
    kFloatType,
    kString,
    kObjectiveType
};


class ConfigValue {
public:
    ConfigValue(const std::string& s, ParamType type) {
        type_ = type;
        if(type_ == kUint32_t) {
            val_int_ = std::stoi(s);
        } else if(type_ == kFloatType) {
            val_float_ = std::stod(s);
        } else if(type_ == kString) {
            val_string_ = s;
        } else if(type_ == kObjectiveType) {
            if(s == "mse") {
                val_objective_ = kMse;
            } else if(s == "logloss") {
                val_objective_ = kLogLoss;
            }
        }
    }

    operator uint32_t() const{
        return val_int_;
    }
    operator float_type() const{
        return val_float_;
    }
    operator std::string() const{
        return val_string_;
    }
    operator ObjectiveType() const{
        return val_objective_;
    }
    ParamType GetType() {
        return type_;
    }

private:
    uint32_t val_int_;
    float_type val_float_;
    std::string val_string_;
    ObjectiveType val_objective_;
    ParamType type_; 
};



#endif //CPP_CONFIGVALUE_H
