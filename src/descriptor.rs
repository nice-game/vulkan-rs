pub use ash::vk::DescriptorType;

use crate::{
	device::Device,
	image::{ImageLayout, ImageView, Sampler},
	shader::ShaderStageFlags,
	sync::Resource,
};
use ash::{version::DeviceV1_0, vk};
use std::{
	marker::PhantomData,
	mem::transmute,
	sync::{Arc, Mutex},
};

pub struct DescriptorSetLayout {
	device: Arc<Device>,
	pub(crate) vk: vk::DescriptorSetLayout,
	_immutable_samplers: Vec<Arc<Sampler>>,
	binding_count: u32,
}
impl DescriptorSetLayout {
	pub fn builder(device: Arc<Device>) -> DescriptorSetLayoutBuilder {
		DescriptorSetLayoutBuilder::new(device)
	}
}
impl Drop for DescriptorSetLayout {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_descriptor_set_layout(self.vk, None) };
	}
}

pub struct DescriptorSetLayoutBuilder {
	device: Arc<Device>,
	bindings: Vec<vk::DescriptorSetLayoutBinding>,
	immutable_samplers: Vec<Arc<Sampler>>,
}
impl DescriptorSetLayoutBuilder {
	fn new(device: Arc<Device>) -> Self {
		Self { device, bindings: vec![], immutable_samplers: vec![] }
	}

	pub fn build(self) -> Arc<DescriptorSetLayout> {
		let ci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&self.bindings);
		let vk = unsafe { self.device.vk.create_descriptor_set_layout(&ci, None) }.unwrap();
		Arc::new(DescriptorSetLayout {
			device: self.device,
			vk,
			_immutable_samplers: self.immutable_samplers,
			binding_count: self.bindings.len() as _,
		})
	}

	pub fn desc(
		mut self,
		descriptor_type: DescriptorType,
		descriptor_count: u32,
		stage_flags: ShaderStageFlags,
		immutable_samplers: impl IntoIterator<Item = Arc<Sampler>>,
	) -> Self {
		let immutable_samplers = immutable_samplers.into_iter();
		let (lower, upper) = immutable_samplers.size_hint();
		let size = upper.unwrap_or(lower);
		let mut immutable_sampler_vks = Vec::with_capacity(size);
		self.immutable_samplers.reserve(size);
		for immutable_sampler in immutable_samplers {
			immutable_sampler_vks.push(immutable_sampler.vk);
			self.immutable_samplers.push(immutable_sampler);
		}

		let binding = vk::DescriptorSetLayoutBinding::builder()
			.descriptor_type(descriptor_type)
			.descriptor_count(descriptor_count)
			.stage_flags(stage_flags)
			.immutable_samplers(&immutable_sampler_vks)
			.build();
		self.bindings.push(binding);
		self
	}
}

pub struct DescriptorPoolSize {
	_vk: vk::DescriptorPoolSize,
}
impl From<(DescriptorType, u32)> for DescriptorPoolSize {
	fn from(val: (DescriptorType, u32)) -> Self {
		Self { _vk: vk::DescriptorPoolSize::builder().ty(val.0).descriptor_count(val.1).build() }
	}
}

pub struct DescriptorPool {
	device: Arc<Device>,
	vk: vk::DescriptorPool,
}
impl DescriptorPool {
	pub fn new(device: Arc<Device>, max_sets: u32, pool_sizes: Vec<DescriptorPoolSize>) -> Arc<Self> {
		let ci = vk::DescriptorPoolCreateInfo::builder()
			.max_sets(max_sets)
			.pool_sizes(unsafe { transmute(&pool_sizes[..]) });
		let vk = unsafe { device.vk.create_descriptor_pool(&ci, None) }.unwrap();
		Arc::new(Self { device, vk })
	}
}
impl Drop for DescriptorPool {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_descriptor_pool(self.vk, None) };
	}
}

pub struct DescriptorSet {
	_descriptor_pool: Arc<DescriptorPool>,
	pub(crate) vk: vk::DescriptorSet,
	// one vec per binding
	resources: Mutex<Vec<Vec<Resource>>>,
}
impl DescriptorSet {
	pub fn alloc(
		descriptor_pool: Arc<DescriptorPool>,
		set_layouts: Vec<Arc<DescriptorSetLayout>>,
	) -> impl Iterator<Item = Arc<DescriptorSet>> {
		let set_layout_vks: Vec<_> = set_layouts.iter().map(|x| x.vk).collect();
		let ci =
			vk::DescriptorSetAllocateInfo::builder().descriptor_pool(descriptor_pool.vk).set_layouts(&set_layout_vks);
		let vks = unsafe { descriptor_pool.device.vk.allocate_descriptor_sets(&ci) }.unwrap();

		vks.into_iter().zip(set_layouts).map(move |(vk, layout)| {
			Arc::new(DescriptorSet {
				_descriptor_pool: descriptor_pool.clone(),
				vk,
				resources: Mutex::new(vec![vec![]; layout.binding_count as _]),
			})
		})
	}

	pub fn update_builder(device: &Device) -> DescriptorSetUpdate {
		DescriptorSetUpdate::new(device)
	}
}

pub struct DescriptorSetUpdate<'a, 'b> {
	device: &'a Device,
	writes: Vec<vk::WriteDescriptorSet>,
	phantom: PhantomData<&'b u8>,
}
impl<'a> DescriptorSetUpdate<'a, 'static> {
	fn new(device: &'a Device) -> Self {
		Self { device, writes: vec![], phantom: PhantomData }
	}
}
impl<'a, 'b> DescriptorSetUpdate<'a, 'b> {
	pub fn write<'c>(
		mut self,
		dst_set: &'b DescriptorSet,
		dst_binding: u32,
		descriptor_type: DescriptorType,
		image_infos: impl IntoIterator<Item = (Option<Arc<Sampler>>, Arc<ImageView>, ImageLayout)>,
	) -> DescriptorSetUpdate<'a, 'c> {
		let image_infos = image_infos.into_iter();
		let (lower, upper) = image_infos.size_hint();
		let size = upper.unwrap_or(lower);

		// TODO: buffer resources, just in case this builder is dropped without being submitted
		let mut resources = dst_set.resources.lock().unwrap();
		let resources = &mut resources[dst_binding as usize];
		resources.clear();
		resources.reserve(size);

		let mut image_info_vks = Vec::with_capacity(size);

		for (sampler, view, layout) in image_infos {
			let info = vk::DescriptorImageInfo::builder()
				.sampler(sampler.as_ref().map(|x| x.vk).unwrap_or(vk::Sampler::null()))
				.image_view(view.vk)
				.image_layout(layout)
				.build();
			image_info_vks.push(info);

			if let Some(sampler) = sampler {
				resources.push(Resource::Sampler(sampler));
			}
			resources.push(Resource::ImageView(view));
		}

		let write = vk::WriteDescriptorSet::builder()
			.dst_set(dst_set.vk)
			.dst_binding(dst_binding)
			.descriptor_type(descriptor_type)
			.image_info(&image_info_vks)
			.build();
		self.writes.push(write);

		DescriptorSetUpdate { device: self.device, writes: self.writes, phantom: PhantomData }
	}

	pub fn submit(self) {
		unsafe { self.device.vk.update_descriptor_sets(&self.writes, &[]) };
	}
}
