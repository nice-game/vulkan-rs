use crate::{pipeline::ComputePipelineBuilder, Instance};
pub use ash::vk::BufferUsageFlags;

use crate::{
	command::CommandBuffer,
	physical_device::{PhysicalDevice, QueueFamily},
	pipeline::{GraphicsPipelineBuilder, PipelineLayout},
	render_pass::RenderPass,
	sync::{GpuFuture, SubmitState},
};
use ash::{
	extensions::khr,
	version::{DeviceV1_0, InstanceV1_0},
	vk, Device as VkDevice,
};
use std::sync::Arc;
use typenum::B0;
use vk_mem::{Allocator, AllocatorCreateInfo};

pub struct Device {
	physical_device: Arc<PhysicalDevice>,
	pub vk: VkDevice,
	pub khr_swapchain: khr::Swapchain,
	pub allocator: Allocator,
}
impl Device {
	// TODO: find a better way to request queues
	pub fn new<'a>(
		physical_device: Arc<PhysicalDevice>,
		qfams: impl IntoIterator<Item = (QueueFamily, &'a [f32])>,
	) -> (Arc<Self>, impl Iterator<Item = Arc<Queue>>) {
		let qcis: Vec<_> = qfams
			.into_iter()
			.inspect(|(qfam, _)| assert!(qfam.physical_device() == &physical_device))
			.map(|(qfam, priorities)| {
				vk::DeviceQueueCreateInfo::builder().queue_family_index(qfam.idx).queue_priorities(priorities).build()
			})
			.collect();

		let exts = [b"VK_KHR_swapchain\0".as_ptr() as _];

		let ci = vk::DeviceCreateInfo::builder().queue_create_infos(&qcis).enabled_extension_names(&exts);
		let vk = unsafe { physical_device.instance().vk.create_device(physical_device.vk, &ci, None) }.unwrap();

		let khr_swapchain = khr::Swapchain::new(&physical_device.instance().vk, &vk);

		let ci = AllocatorCreateInfo {
			physical_device: physical_device.vk,
			device: vk.clone(),
			instance: physical_device.instance().vk.clone(),
			..AllocatorCreateInfo::default()
		};
		let allocator = Allocator::new(&ci).unwrap();

		let device = Arc::new(Self { physical_device, vk, khr_swapchain, allocator });

		let device2 = device.clone();
		let queues = qcis
			.into_iter()
			.map(move |qci| {
				let device = device2.clone();
				(0..qci.queue_count).map(move |idx| unsafe { device.get_queue(qci.queue_family_index, idx) })
			})
			.flatten();

		(device, queues)
	}

	pub fn build_compute_pipeline(self: &Arc<Self>, layout: Arc<PipelineLayout>) -> ComputePipelineBuilder {
		ComputePipelineBuilder::new(self.clone(), layout)
	}

	pub fn build_graphics_pipeline(
		self: &Arc<Self>,
		layout: Arc<PipelineLayout>,
		render_pass: Arc<RenderPass>,
	) -> GraphicsPipelineBuilder<'static, ()> {
		GraphicsPipelineBuilder::new(self.clone(), layout, render_pass)
	}

	pub fn instance(&self) -> &Arc<Instance> {
		self.physical_device.instance()
	}

	pub fn physical_device(&self) -> &Arc<PhysicalDevice> {
		&self.physical_device
	}

	pub(crate) unsafe fn get_queue(self: &Arc<Self>, queue_family_index: u32, queue_index: u32) -> Arc<Queue> {
		let vk = self.vk.get_device_queue(queue_family_index, queue_index);

		Arc::new(Queue {
			device: self.clone(),
			family: QueueFamily::from_vk(self.physical_device.clone(), queue_family_index),
			vk,
		})
	}
}
impl Drop for Device {
	fn drop(&mut self) {
		self.allocator.destroy();
		unsafe { self.vk.destroy_device(None) };
	}
}

pub struct Queue {
	pub(crate) device: Arc<Device>,
	family: QueueFamily,
	pub vk: vk::Queue,
}
impl Queue {
	pub fn device(&self) -> &Arc<Device> {
		&self.device
	}

	pub fn family(&self) -> &QueueFamily {
		&self.family
	}

	pub fn submit(self: &Arc<Self>, cmd: Arc<CommandBuffer<B0>>) -> SubmitFuture {
		assert!(cmd.pool.queue_family == self.family);
		SubmitFuture { queue: self.clone(), cmd }
	}

	pub fn submit_after<T: GpuFuture>(self: &Arc<Self>, prev: T, cmd: Arc<CommandBuffer<B0>>) -> SubmitAfterFuture<T> {
		assert!(cmd.pool.queue_family == self.family);
		SubmitAfterFuture { queue: self.clone(), cmd, prev }
	}
}
impl PartialEq for Queue {
	fn eq(&self, other: &Self) -> bool {
		self.vk == other.vk
	}
}
impl Eq for Queue {}

pub struct SubmitFuture {
	queue: Arc<Queue>,
	cmd: Arc<CommandBuffer<B0>>,
}
impl GpuFuture for SubmitFuture {
	fn build_submission(&mut self) -> SubmitState {
		let mut submit = SubmitState::new();
		submit.cmd(&self.cmd);
		submit
	}

	fn device(&self) -> &Arc<Device> {
		self.queue.device()
	}

	fn flush(&mut self) {
		todo!()
	}

	fn queue(&self) -> Option<&Arc<Queue>> {
		Some(&self.queue)
	}
}
// impl SubmitFuture {
// 	pub fn end(self) -> Fence {
// 		let fence = Fence::new(self.queue.device.clone(), false, vec![self.cmd.clone()]);

// 		let submits = [vk::SubmitInfo::builder().command_buffers(&[self.cmd.vk]).build()];
// 		unsafe { self.queue.device().vk.queue_submit(self.queue.vk, &submits, fence.vk) }.unwrap();

// 		fence
// 	}
// }

pub struct SubmitAfterFuture<T> {
	queue: Arc<Queue>,
	cmd: Arc<CommandBuffer<B0>>,
	prev: T,
}
impl<T: GpuFuture> GpuFuture for SubmitAfterFuture<T> {
	fn build_submission(&mut self) -> SubmitState {
		let mut submit = self.prev.build_submission();
		submit.cmd(&self.cmd);
		submit
	}

	fn device(&self) -> &Arc<Device> {
		self.queue.device()
	}

	fn flush(&mut self) {
		todo!()
	}

	fn queue(&self) -> Option<&Arc<Queue>> {
		Some(&self.queue)
	}
}
// impl<T: GpuFuture> SubmitAfterFuture<T> {
// 	pub fn end(self) -> Fence {
// 		let (semaphores, stages) = self.prev.semaphores();
// 		let mut resources = Vec::with_capacity(semaphores.len() + 1);
// 		let mut semaphore_vks = Vec::with_capacity(semaphores.len());
// 		for semaphore in semaphores {
// 			semaphore_vks.push(semaphore.vk);
// 			resources.push(Resource::Semaphore(semaphore));
// 		}

// 		let fence = Fence::new(self.queue.device.clone(), false, vec![self.cmd.clone()]);

// 		let submits = [vk::SubmitInfo::builder()
// 			.wait_semaphores(&semaphore_vks)
// 			.wait_dst_stage_mask(&stages)
// 			.command_buffers(&[self.cmd.vk])
// 			.build()];
// 		unsafe { self.queue.device().vk.queue_submit(self.queue.vk, &submits, fence.vk) }.unwrap();

// 		fence
// 	}

// 	pub fn flush(self) -> (Fence, FlushFuture) {
// 		let (semaphores, stages) = self.prev.semaphores();
// 		let mut resources = Vec::with_capacity(semaphores.len() + 1);
// 		let mut semaphore_vks = Vec::with_capacity(semaphores.len());
// 		for semaphore in semaphores {
// 			semaphore_vks.push(semaphore.vk);
// 			resources.push(Resource::Semaphore(semaphore));
// 		}

// 		let fence = Fence::new(self.queue.device.clone(), false, vec![self.cmd.clone()]);
// 		let semaphore = Semaphore::new(self.queue.device.clone());

// 		let submits = [vk::SubmitInfo::builder()
// 			.wait_semaphores(&semaphore_vks)
// 			.wait_dst_stage_mask(&stages)
// 			.command_buffers(&[self.cmd.vk])
// 			.signal_semaphores(&[semaphore.vk])
// 			.build()];
// 		unsafe { self.queue.device().vk.queue_submit(self.queue.vk, &submits, fence.vk) }.unwrap();

// 		(fence, FlushFuture { semaphore })
// 	}
// }
