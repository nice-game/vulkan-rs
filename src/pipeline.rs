pub use ash::vk::{VertexInputAttributeDescription, Viewport};

use crate::{
	descriptor::DescriptorSetLayout,
	device::Device,
	render_pass::RenderPass,
	shader::{ShaderModule, ShaderStageFlags},
	Extent2D, Offset2D,
};
use ash::{version::DeviceV1_0, vk};
use std::{
	ffi::CStr,
	marker::PhantomData,
	mem::{size_of, transmute},
	sync::Arc,
};

pub struct PipelineLayout {
	device: Arc<Device>,
	pub vk: vk::PipelineLayout,
	_set_layouts: Vec<Arc<DescriptorSetLayout>>,
}
impl PipelineLayout {
	pub fn new(
		device: Arc<Device>,
		set_layouts: Vec<Arc<DescriptorSetLayout>>,
		push_constant_ranges: impl IntoIterator<Item = (ShaderStageFlags, u32, u32)>,
	) -> Arc<PipelineLayout> {
		let set_layout_vks: Vec<_> = set_layouts.iter().map(|x| x.vk).collect();

		let push_constant_ranges: Vec<_> = push_constant_ranges
			.into_iter()
			.map(|(stage_flags, offset, size)| {
				vk::PushConstantRange::builder().stage_flags(stage_flags).offset(offset).size(size).build()
			})
			.collect();

		let ci = vk::PipelineLayoutCreateInfo::builder()
			.set_layouts(&set_layout_vks)
			.push_constant_ranges(&push_constant_ranges)
			.build();
		let vk = unsafe { device.vk.create_pipeline_layout(&ci, None) }.unwrap();
		Arc::new(Self { device, vk, _set_layouts: set_layouts })
	}
}
impl Drop for PipelineLayout {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_pipeline_layout(self.vk, None) };
	}
}

pub struct ComputePipeline {
	device: Arc<Device>,
	_layout: Arc<PipelineLayout>,
	_shader: Arc<ShaderModule>,
	pub vk: vk::Pipeline,
}
impl Drop for ComputePipeline {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_pipeline(self.vk, None) };
	}
}

pub struct GraphicsPipeline {
	device: Arc<Device>,
	_layout: Arc<PipelineLayout>,
	render_pass: Arc<RenderPass>,
	_vertex_shader: Arc<ShaderModule>,
	_fragment_shader: Arc<ShaderModule>,
	pub vk: vk::Pipeline,
}
impl GraphicsPipeline {
	pub fn render_pass(&self) -> &Arc<RenderPass> {
		&self.render_pass
	}
}
impl Drop for GraphicsPipeline {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_pipeline(self.vk, None) };
	}
}

pub struct GraphicsPipelineBuilder<'a, T: VertexDesc> {
	device: Arc<Device>,
	layout: Arc<PipelineLayout>,
	render_pass: Arc<RenderPass>,
	vertex_shader: Option<Arc<ShaderModule>>,
	fragment_shader: Option<Arc<ShaderModule>>,
	vertex_input: PhantomData<T>,
	viewports: &'a [Viewport],
}
impl<'a, T: VertexDesc> GraphicsPipelineBuilder<'a, T> {
	pub fn build(self) -> Arc<GraphicsPipeline> {
		let mut stages = vec![
			vk::PipelineShaderStageCreateInfo::builder()
				.stage(vk::ShaderStageFlags::VERTEX)
				.module(self.vertex_shader.as_ref().unwrap().vk)
				.name(CStr::from_bytes_with_nul(b"main\0").unwrap())
				.build(),
		];
		if let Some(fragment_shader) = &self.fragment_shader {
			stages.push(
				vk::PipelineShaderStageCreateInfo::builder()
					.stage(vk::ShaderStageFlags::FRAGMENT)
					.module(fragment_shader.vk)
					.name(CStr::from_bytes_with_nul(b"main\0").unwrap())
					.build(),
			);
		}

		let vertex_binding_descriptions = [vk::VertexInputBindingDescription::builder()
			.binding(0)
			.stride(size_of::<T>() as _)
			.input_rate(vk::VertexInputRate::VERTEX)
			.build()];
		let vertex_attribute_descriptions = T::attribute_descs();
		let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
			.vertex_binding_descriptions(&vertex_binding_descriptions)
			.vertex_attribute_descriptions(&vertex_attribute_descriptions);
		let input_assembly_state =
			vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
		let scissors: Vec<_> = self
			.viewports
			.iter()
			.map(|v| {
				vk::Rect2D::builder()
					.offset(Offset2D { x: v.x as _, y: v.y as _ })
					.extent(Extent2D { width: v.width as _, height: v.height as _ })
					.build()
			})
			.collect();
		let viewport_state =
			vk::PipelineViewportStateCreateInfo::builder().viewports(self.viewports).scissors(&scissors);
		let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
			.polygon_mode(vk::PolygonMode::FILL)
			.cull_mode(vk::CullModeFlags::BACK)
			.front_face(vk::FrontFace::CLOCKWISE)
			.line_width(1.0);
		let multisample_state =
			vk::PipelineMultisampleStateCreateInfo::builder().rasterization_samples(vk::SampleCountFlags::TYPE_1);
		let attachments =
			[vk::PipelineColorBlendAttachmentState::builder().color_write_mask(vk::ColorComponentFlags::all()).build()];
		let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&attachments);
		let cis = [vk::GraphicsPipelineCreateInfo::builder()
			.stages(&stages)
			.vertex_input_state(&vertex_input_state)
			.input_assembly_state(&input_assembly_state)
			.viewport_state(&viewport_state)
			.rasterization_state(&rasterization_state)
			.multisample_state(&multisample_state)
			.color_blend_state(&color_blend_state)
			.layout(self.layout.vk)
			.render_pass(self.render_pass.vk)
			.build()];
		let vk = unsafe { self.device.vk.create_graphics_pipelines(vk::PipelineCache::null(), &cis, None) }.unwrap()[0];

		Arc::new(GraphicsPipeline {
			device: self.device,
			_layout: self.layout,
			render_pass: self.render_pass,
			_vertex_shader: self.vertex_shader.unwrap(),
			_fragment_shader: self.fragment_shader.unwrap(),
			vk,
		})
	}

	pub fn vertex_shader(mut self, vertex_shader: Arc<ShaderModule>) -> Self {
		self.vertex_shader = Some(vertex_shader);
		self
	}

	pub fn fragment_shader(mut self, fragment_shader: Arc<ShaderModule>) -> Self {
		self.fragment_shader = Some(fragment_shader);
		self
	}

	pub fn vertex_input<V: VertexDesc>(self) -> GraphicsPipelineBuilder<'a, V> {
		unsafe { transmute(self) }
	}

	pub fn viewports<'b>(self, viewports: &'b [Viewport]) -> GraphicsPipelineBuilder<'b, T> {
		let mut this: GraphicsPipelineBuilder<'b, T> = unsafe { transmute(self) };
		this.viewports = viewports;
		this
	}

	pub(crate) fn new(device: Arc<Device>, layout: Arc<PipelineLayout>, render_pass: Arc<RenderPass>) -> Self {
		Self {
			device,
			layout,
			render_pass,
			vertex_shader: None,
			fragment_shader: None,
			vertex_input: PhantomData,
			viewports: &[],
		}
	}
}

pub struct ComputePipelineBuilder {
	device: Arc<Device>,
	layout: Arc<PipelineLayout>,
	shader: Option<Arc<ShaderModule>>,
}
impl ComputePipelineBuilder {
	pub fn build(self) -> Arc<ComputePipeline> {
		let stage = vk::PipelineShaderStageCreateInfo::builder()
			.stage(vk::ShaderStageFlags::VERTEX)
			.module(self.shader.as_ref().unwrap().vk)
			.name(CStr::from_bytes_with_nul(b"main\0").unwrap())
			.build();

		let cis = [vk::ComputePipelineCreateInfo::builder().stage(stage).layout(self.layout.vk).build()];
		let vk = unsafe { self.device.vk.create_compute_pipelines(vk::PipelineCache::null(), &cis, None) }.unwrap()[0];

		Arc::new(ComputePipeline { device: self.device, _layout: self.layout, _shader: self.shader.unwrap(), vk })
	}

	pub fn shader(mut self, shader: Arc<ShaderModule>) -> Self {
		self.shader = Some(shader);
		self
	}

	pub(crate) fn new(device: Arc<Device>, layout: Arc<PipelineLayout>) -> Self {
		Self { device, layout, shader: None }
	}
}

pub trait VertexDesc {
	fn attribute_descs() -> Vec<VertexInputAttributeDescription>;
}
impl VertexDesc for () {
	fn attribute_descs() -> Vec<VertexInputAttributeDescription> {
		vec![]
	}
}
