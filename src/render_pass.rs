pub use ash::vk::{
	AccessFlags, AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, PipelineBindPoint,
	SampleCountFlags, SubpassDependency, SubpassDescription, SUBPASS_EXTERNAL,
};

use crate::device::Device;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

pub struct RenderPass {
	device: Arc<Device>,
	pub vk: vk::RenderPass,
}
impl RenderPass {
	pub fn new(
		device: &Arc<Device>,
		attachments: &[AttachmentDescription],
		subpasses: &[SubpassDescription],
		dependencies: &[SubpassDependency],
	) -> Arc<Self> {
		let ci = vk::RenderPassCreateInfo::builder()
			.attachments(&attachments)
			.subpasses(&subpasses)
			.dependencies(&dependencies);
		let vk = unsafe { device.vk.create_render_pass(&ci, None) }.unwrap();
		Arc::new(Self { device: device.clone(), vk })
	}

	pub fn device(&self) -> &Arc<Device> {
		&self.device
	}

	pub unsafe fn from_vk(device: Arc<Device>, vk: vk::RenderPass) -> Arc<Self> {
		Arc::new(Self { device, vk })
	}
}
impl Drop for RenderPass {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_render_pass(self.vk, None) };
	}
}

#[macro_export]
#[rustfmt::skip]
macro_rules! ordered_passes_renderpass {
	(
		$device:expr,
		attachments: {
			$(
				$atch_name:ident : {
					load: $load:ident,
					store: $store:ident,
					format: $format:expr,
					samples: $samples:expr,
					$(initial_layout: $init_layout:expr,)*
					$(final_layout: $final_layout:expr,)*
				}
			),*
		},
		passes: [
			$(
				{
					color: [$($color_atch:ident),*],
					depth_stencil: { $($depth_atch:ident)* },
					input: [$($input_atch:ident),*] $(,)*
					$(resolve: [$($resolve_atch:ident),*])* $(,)*
				}
			),*
		]
	) => {{
		let attachments = [$(
			$crate::render_pass::AttachmentDescription::builder()
				.format($format)
				.samples($crate::render_pass::SampleCountFlags::TYPE_1)
				.load_op($crate::render_pass::AttachmentLoadOp::CLEAR)
				.store_op($crate::render_pass::AttachmentStoreOp::STORE)
				.stencil_load_op($crate::render_pass::AttachmentLoadOp::DONT_CARE)
				.stencil_store_op($crate::render_pass::AttachmentStoreOp::DONT_CARE)
				.initial_layout($crate::image::ImageLayout::UNDEFINED)
				.final_layout($crate::image::ImageLayout::PRESENT_SRC_KHR)
				.build()
		),*];
		let color_attachments =
			[$crate::render_pass::AttachmentReference::builder().layout($crate::image::ImageLayout::COLOR_ATTACHMENT_OPTIMAL).build()];
		let subpasses = [$crate::render_pass::SubpassDescription::builder()
			.pipeline_bind_point($crate::render_pass::PipelineBindPoint::GRAPHICS)
			.color_attachments(&color_attachments)
			.build()];
		let dependencies = [$crate::render_pass::SubpassDependency::builder()
			.src_subpass($crate::render_pass::SUBPASS_EXTERNAL)
			.src_stage_mask($crate::command::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
			.dst_stage_mask($crate::command::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
			.dst_access_mask($crate::render_pass::AccessFlags::COLOR_ATTACHMENT_READ | $crate::render_pass::AccessFlags::COLOR_ATTACHMENT_WRITE)
			.build()];
		$crate::render_pass::RenderPass::new($device, &attachments, &subpasses, &dependencies)
	}};
}
