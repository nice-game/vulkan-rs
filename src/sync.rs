use crate::{
	buffer::BufferAbstract,
	command::CommandBuffer,
	descriptor::DescriptorSet,
	device::{Device, Queue},
	image::{Framebuffer, Image, ImageView, Sampler},
	pipeline::{GraphicsPipeline, PipelineLayout},
	render_pass::RenderPass,
};
use ash::{version::DeviceV1_0, vk};
use crossbeam::atomic::AtomicCell;
use std::sync::Arc;
use typenum::{B0, B1};

pub struct Semaphore {
	pub(crate) device: Arc<Device>,
	pub vk: vk::Semaphore,
}
impl Semaphore {
	pub(crate) fn new(device: Arc<Device>) -> Arc<Semaphore> {
		unsafe {
			let vk = device.vk.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None).unwrap();
			Arc::new(Self { device, vk })
		}
	}
}
impl Drop for Semaphore {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_semaphore(self.vk, None) };
	}
}

pub trait GpuFuture {
	fn build_submission(&mut self) -> SubmitState;
	fn device(&self) -> &Arc<Device>;
	fn flush(&mut self);
	fn queue(&self) -> Option<&Arc<Queue>>;

	fn join<R: GpuFuture>(self, other: R) -> JoinFuture<Self, R>
	where
		Self: Sized,
	{
		JoinFuture::new(self, other)
	}

	fn then_signal_fence(self) -> Fence
	where
		Self: Send + Sync + Sized + 'static,
	{
		Fence::end(self)
	}

	fn then_signal_semaphore(self) -> (SemaphoreSignalFuture<Self>, SemaphoreFuture)
	where
		Self: Sized,
	{
		let semaphore = Semaphore::new(self.device().clone());
		(SemaphoreSignalFuture::new(self, semaphore.clone()), SemaphoreFuture::new(semaphore))
	}
}
impl GpuFuture for Box<dyn GpuFuture> {
	fn build_submission(&mut self) -> SubmitState {
		(**self).build_submission()
	}

	fn device(&self) -> &Arc<Device> {
		(**self).device()
	}

	fn flush(&mut self) {
		(**self).flush()
	}

	fn queue(&self) -> Option<&Arc<Queue>> {
		(**self).queue()
	}
}

pub struct SubmitState {
	wait_semaphores: Vec<vk::Semaphore>,
	wait_dst_stage_masks: Vec<vk::PipelineStageFlags>,
	signal_semaphores: Vec<vk::Semaphore>,
	cmds: Vec<vk::CommandBuffer>,
}
impl SubmitState {
	pub(crate) fn new() -> Self {
		Self { wait_semaphores: vec![], wait_dst_stage_masks: vec![], signal_semaphores: vec![], cmds: vec![] }
	}

	pub(crate) fn wait_semaphore(&mut self, semaphore: &Semaphore, wait_dst_stage_mask: vk::PipelineStageFlags) {
		self.wait_semaphores.push(semaphore.vk);
		self.wait_dst_stage_masks.push(wait_dst_stage_mask);
	}

	pub(crate) fn signal_semaphore(&mut self, semaphore: &Semaphore) {
		self.signal_semaphores.push(semaphore.vk);
	}

	pub(crate) fn cmd(&mut self, cmd: &CommandBuffer<B0>) {
		self.cmds.push(cmd.vk);
	}

	fn join(&mut self, other: SubmitState) {
		self.wait_semaphores.extend(other.wait_semaphores);
		self.wait_dst_stage_masks.extend(other.wait_dst_stage_masks);
		self.signal_semaphores.extend(other.signal_semaphores);
		self.cmds.extend(other.cmds);
	}
}

pub struct SemaphoreFuture {
	semaphore: Arc<Semaphore>,
}
impl SemaphoreFuture {
	pub fn new(semaphore: Arc<Semaphore>) -> Self {
		Self { semaphore }
	}

	pub(crate) fn semaphore(&self) -> &Arc<Semaphore> {
		&self.semaphore
	}
}
impl GpuFuture for SemaphoreFuture {
	fn build_submission(&mut self) -> SubmitState {
		todo!()
	}

	fn device(&self) -> &Arc<Device> {
		&self.semaphore.device
	}

	fn flush(&mut self) {
		todo!()
	}

	fn queue(&self) -> Option<&Arc<Queue>> {
		None
	}
}

pub struct SemaphoreSignalFuture<P> {
	prev: P,
	semaphore: Arc<Semaphore>,
}
impl<P: GpuFuture> SemaphoreSignalFuture<P> {
	pub fn new(prev: P, semaphore: Arc<Semaphore>) -> Self {
		Self { prev, semaphore }
	}
}
impl<P: GpuFuture> GpuFuture for SemaphoreSignalFuture<P> {
	fn build_submission(&mut self) -> SubmitState {
		let mut submit = self.prev.build_submission();
		submit.signal_semaphore(&self.semaphore);
		submit
	}

	fn device(&self) -> &Arc<Device> {
		self.prev.device()
	}

	fn flush(&mut self) {
		todo!()
	}

	fn queue(&self) -> Option<&Arc<Queue>> {
		self.prev.queue()
	}
}

pub struct Fence {
	device: Arc<Device>,
	prev: AtomicCell<Option<Box<dyn GpuFuture + Send + Sync>>>,
	vk: vk::Fence,
}
impl Fence {
	pub fn new(device: &Arc<Device>, signalled: bool) -> Self {
		let flags = if signalled { vk::FenceCreateFlags::SIGNALED } else { vk::FenceCreateFlags::empty() };
		let vk = unsafe { device.vk.create_fence(&vk::FenceCreateInfo::builder().flags(flags), None) }.unwrap();
		Self { device: device.clone(), prev: AtomicCell::default(), vk }
	}

	pub fn end(mut prev: impl GpuFuture + Send + Sync + 'static) -> Self {
		let submit = prev.build_submission();

		let vk = unsafe { prev.device().vk.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();

		let submits = [vk::SubmitInfo::builder()
			.wait_semaphores(&submit.wait_semaphores)
			.wait_dst_stage_mask(&submit.wait_dst_stage_masks)
			.signal_semaphores(&submit.signal_semaphores)
			.command_buffers(&submit.cmds)
			.build()];
		unsafe { prev.device().vk.queue_submit(prev.queue().unwrap().vk, &submits, vk) }.unwrap();

		Self { device: prev.device().clone(), prev: AtomicCell::new(Some(Box::new(prev))), vk }
	}

	pub fn wait(&self) {
		unsafe { self.device.vk.wait_for_fences(&[self.vk], false, !0) }.unwrap();
		self.prev.take();
	}
}
impl Drop for Fence {
	fn drop(&mut self) {
		self.wait();
		unsafe { self.device.vk.destroy_fence(self.vk, None) };
	}
}

pub struct JoinFuture<L, R> {
	left: L,
	right: R,
}
impl<L: GpuFuture, R: GpuFuture> JoinFuture<L, R> {
	pub fn new(left: L, right: R) -> Self {
		if let Some(lqueue) = left.queue() {
			if let Some(rqueue) = right.queue() {
				assert!(lqueue == rqueue);
			}
		}
		Self { left, right }
	}
}
impl<L: GpuFuture, R: GpuFuture> GpuFuture for JoinFuture<L, R> {
	fn build_submission(&mut self) -> SubmitState {
		let mut submit = self.left.build_submission();
		submit.join(self.right.build_submission());
		submit
	}

	fn device(&self) -> &Arc<Device> {
		self.left.device()
	}

	fn flush(&mut self) {
		self.left.flush();
		self.right.flush();
	}

	fn queue(&self) -> Option<&Arc<Queue>> {
		self.left.queue()
	}
}

pub struct NowFuture {
	device: Arc<Device>,
}
impl NowFuture {
	pub fn new(device: Arc<Device>) -> Self {
		Self { device }
	}
}
impl GpuFuture for NowFuture {
	fn build_submission(&mut self) -> SubmitState {
		SubmitState::new()
	}

	fn device(&self) -> &Arc<Device> {
		&self.device
	}

	fn flush(&mut self) {}

	fn queue(&self) -> Option<&Arc<Queue>> {
		None
	}
}

#[derive(Clone)]
pub(crate) enum Resource {
	Buffer(Arc<dyn BufferAbstract + Send + Sync>),
	// TODO: merge with CommandBufferAbstract trait?
	CommandBufferSecondary(Arc<CommandBuffer<B1>>),
	DescriptorSet(Arc<DescriptorSet>),
	Framebuffer(Arc<Framebuffer>),
	Image(Arc<Image>),
	ImageView(Arc<ImageView>),
	Pipeline(Arc<GraphicsPipeline>),
	PipelineLayout(Arc<PipelineLayout>),
	RenderPass(Arc<RenderPass>),
	Sampler(Arc<Sampler>),
}
