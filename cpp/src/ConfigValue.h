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
    ConfigValue(const std::string& s, ParamType type) {
        type_ = type;
        if(type_ == kUint32_t) {
            val_int_ = std::stoi(s);
        } else if(type_ == kFloatType) {
            val_float_ = std::stod(s);
        } else if(type_ == kString) {
            val_string_ = s;
        } else if(type_ == kBool) {
            val_bool_ = StringToBool(s);
        } else if(type_ == kObjectiveType) {
            val_objective_ = StringToObj(s);
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
    operator bool() const{
        return val_bool_;
    }
    ParamType GetType() {
        return type_;
    }

private:
    ObjectiveType StringToObj(const std::string& s) {
        if(s == "mse") {
            return kMse;
        } else if(s == "logloss") {
            return kLogLoss;
        } else {
             throw std::invalid_argument("Wrong argument for objective");
        }
    };
    bool StringToBool(const std::string& s) {
        if(s == "true") {
            return true;
        } else if(s == "false") {
            return false;
        } else {
             throw std::invalid_argument("Ivalid argument for file_has_target");
        }
    };

    uint32_t val_int_;
    float_type val_float_;
    std::string val_string_;
    ObjectiveType val_objective_;
    bool val_bool_;
    ParamType type_; 
};



#endif //CPP_CONFIGVALUE_H
