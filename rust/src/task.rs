use serde::Deserialize;

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TaskKind {
    #[default]
    Knowledge,
    Coding,
    Math,
    InstructionFollowing,
    FunctionCalling,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AnswerCheckMode {
    SingleLabel,
    LabelSet,
    ExactText,
    JsonExact,
    Disabled,
}

impl TaskKind {
    pub(crate) fn default_answer_check(self) -> AnswerCheckMode {
        match self {
            Self::Knowledge | Self::Math => AnswerCheckMode::ExactText,
            Self::FunctionCalling => AnswerCheckMode::JsonExact,
            Self::Coding | Self::InstructionFollowing => AnswerCheckMode::Disabled,
        }
    }

    pub(crate) fn validates_choice_options(self) -> bool {
        matches!(self, Self::Knowledge)
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Knowledge => "knowledge",
            Self::Coding => "coding",
            Self::Math => "math",
            Self::InstructionFollowing => "instruction_following",
            Self::FunctionCalling => "function_calling",
        }
    }
}

impl AnswerCheckMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::SingleLabel => "single_label",
            Self::LabelSet => "label_set",
            Self::ExactText => "exact_text",
            Self::JsonExact => "json_exact",
            Self::Disabled => "disabled",
        }
    }
}
