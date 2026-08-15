use anyhow::Result;
use mainarch_core::model_api::prelude::*;

struct UnsupportedCollectivePlugin;

impl ModelDefinition for UnsupportedCollectivePlugin {
    fn name(&self) -> &str {
        "unsupported-collective-plugin"
    }

    fn define(&self, api: &mut dyn ModelPrimitiveApi) -> Result<()> {
        api.declare_tensor(TensorSpec::new(
            "expert_input",
            DType::F16,
            vec![16],
            TensorRole::Activation,
        ))?;
        api.declare_tensor(TensorSpec::new(
            "expert_output",
            DType::F16,
            vec![16],
            TensorRole::Activation,
        ))?;

        api.begin_stage("expert_parallel", ModelStageKind::Moe)?;
        api.emit(PrimitiveOp::Collective(Collective {
            name: "ep_all_to_all".to_string(),
            kind: CollectiveKind::AllToAll,
            input: "expert_input".into(),
            output: "expert_output".into(),
            group_size: 2,
        }))?;
        api.end_stage()?;

        Ok(())
    }
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let emit_rejection_receipt = args.iter().any(|arg| arg == "--rejection-receipt");

    let model = UnsupportedCollectivePlugin;
    let catalog = MainarchPrimitiveLoweringCatalog::mi355_reference();
    let inspection = inspect_model_plugin(&model, &catalog)?;
    let rejection = inspection.rejection_report();
    rejection.assert_consistent_with(&inspection)?;
    rejection.assert_rejected()?;

    if emit_rejection_receipt {
        print!("{}", rejection.receipt_text());
        return Ok(());
    }

    println!("model: {}", rejection.summary.model_name);
    println!(
        "plugin_rejection: receipt_fingerprint={} rejected={} issues={} readiness_issues={} compatibility_issues={} lowering_gaps={} stage_gaps={} unstaged_ops={} missing_checkpoint_weights={} binding_issues={}",
        rejection.receipt_fingerprint(),
        rejection.is_rejected(),
        rejection.rejection_issue_count,
        rejection.readiness_issues.issues.len(),
        rejection.compatibility_issues.len(),
        rejection.lowering_gap_op_names.len(),
        rejection.stage_gap_names.len(),
        rejection.unstaged_op_names.len(),
        rejection.missing_checkpoint_weight_names.len(),
        rejection.binding_issue_tensor_names.len()
    );
    for issue in &rejection.readiness_issues.issues {
        println!(
            "readiness_issue: kind={} surface={} subject={} message={}",
            issue.kind.as_str(),
            issue.surface,
            issue.subject,
            issue.message
        );
    }
    for issue in &rejection.compatibility_issues {
        println!(
            "compatibility_issue: kind={} field={} message={}",
            issue.kind.as_str(),
            issue.field,
            issue.message
        );
    }

    Ok(())
}
